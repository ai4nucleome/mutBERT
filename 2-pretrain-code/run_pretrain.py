import argparse
import logging
import math
import os
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from modeling_mutbert import RoPEBertConfig, RoPEBertForMaskedLM
from mutdataset import DNADatasetOH
import utils

logger = get_logger(__name__)


class MyCollateFn:
    def __init__(self, tokenizer, mlm_probability, vocab_size):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.vocab_size = vocab_size

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)  # (batch_size, l, V)
        attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)  # (batch_size, l)
        bs, length = attention_mask.shape
        special_tokens_mask = torch.zeros((bs, length), dtype=torch.int64)
        special_tokens_mask[:, 0] = 1
        special_tokens_mask[:, -1] = 1
        new_batch = {
            'input_ids': input_ids,  # (B, L, V)
            'attention_mask': attention_mask,  # (B, L)
        }

        new_batch["input_ids"], new_batch["labels"], new_batch["masked_indices"] = self.torch_mask_tokens(
            inputs=new_batch["input_ids"],
            special_tokens_mask=special_tokens_mask
        )

        return new_batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        # inputs: (bs, length, vocab_size)
        labels = inputs.clone() 
        bs, seq_len, _ = inputs.shape

        masked_indices = torch.zeros((bs, seq_len), dtype=torch.bool)

        special_tokens_mask = special_tokens_mask.bool()

        is_one_hot = (inputs.count_nonzero(dim=-1) == 1) 
        is_prob = ~is_one_hot 

        valid_mask = ~special_tokens_mask  # (bs, seq_len)
        # [False True True ... True False]

        for b_idx in range(bs):
            prob_positions = (is_prob[b_idx] & valid_mask[b_idx]).nonzero(as_tuple=True)[0]
            oh_positions = (is_one_hot[b_idx] & valid_mask[b_idx]).nonzero(as_tuple=True)[0]
            target_num = int((seq_len - 2) * self.mlm_probability)

            p = prob_positions.size(0)

            if p >= target_num:
                chosen_idx = np.random.choice(p, size=target_num, replace=False)
                chosen_positions = prob_positions[chosen_idx]
            else:
                chosen_positions = prob_positions

                remain = target_num - p
                oh_count = oh_positions.size(0)
                remain = min(remain, oh_count)
                chosen_idx2 = np.random.choice(oh_count, size=remain, replace=False)
                supplement_positions = oh_positions[chosen_idx2]

                chosen_positions = torch.cat([chosen_positions, supplement_positions], dim=0)

            masked_indices[b_idx, chosen_positions] = True


        # 1) 80% mask
        indices_replaced = (torch.bernoulli(torch.full((bs, seq_len), 0.8)).bool()) & masked_indices

        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        mask_token_oh = F.one_hot(torch.tensor(mask_token_id), num_classes=self.vocab_size).float()
        inputs[indices_replaced] = mask_token_oh

        # 2) 10% random
        # random_indices = torch.bernoulli(torch.full((bs, seq_len), 0.5)).bool() & masked_indices & ~indices_replaced
        # rand_word_ids = torch.randint(len(self.tokenizer), (bs, seq_len), dtype=torch.long)
        # rand_words_oh = F.one_hot(rand_word_ids, num_classes=self.vocab_size).float()
        # inputs[random_indices] = rand_words_oh[random_indices]

        return inputs, labels, masked_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Training a transformers model on a Masked Language Modeling task")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where your custom dataset files are located.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Directory for a model",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="mlm probability",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=10000,
        help="Total number of training samples",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10000,
        help="Total number of testing samples",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature to change distribution of input",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay to use.")

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
        help="experiment name for a training",
    )

    args = parser.parse_args()

    # Sanity checks
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"`data_dir` must be a valid directory. Received: {args.data_dir}")

    if args.output_dir is None:
        raise ValueError("`output_dir` must be specified to save the trained model.")

    return args


def set_seed(seed, accelerator):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    seed += accelerator.process_index
    torch.manual_seed(seed)

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {"log_with": "tensorboard", "project_dir": args.output_dir}
    accelerator = Accelerator(mixed_precision="fp16", **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        # use DDP, each device has a specific seed
        set_seed(args.seed, accelerator=accelerator)

    # create output dir
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
    accelerator.wait_for_everyone()

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Initialize custom datasets
    train_dataset = DNADatasetOH(data_path=args.data_dir, seq_length=args.max_seq_length-2,
                                 vocab_size=len(tokenizer), split="train",
                                 total=args.train_samples, temperature=args.temperature)
    eval_dataset = DNADatasetOH(data_path=args.data_dir, seq_length=args.max_seq_length-2,
                                 vocab_size=len(tokenizer), split="test",
                                 total=args.test_samples, temperature=args.temperature)

    # Initialize your custom config and model
    config = RoPEBertConfig.from_pretrained(args.model_name_or_path)
    logger.info("Training new model from scratch")
    model = RoPEBertForMaskedLM(config=config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(config)
    accelerator.print("Model Parameters", n_parameters)

    # Data collator
    data_collator = MyCollateFn(tokenizer=tokenizer,
                                mlm_probability=args.mlm_probability,
                                vocab_size=len(tokenizer))

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=16,
        drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=16
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    num_update_steps_per_epoch = len(train_dataset) // total_batch_size

    if args.max_train_steps % num_update_steps_per_epoch == 0:
        args.num_train_epochs = int(args.max_train_steps / num_update_steps_per_epoch)
    else:
        args.num_train_epochs = int(args.max_train_steps / num_update_steps_per_epoch) + 1

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = utils.WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.num_warmup_steps,
        total_steps=args.max_train_steps,
        base_lr=args.learning_rate,
        init_lr=1e-6,
        min_lr=1e-8
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    tb_log_dir = os.path.join("tb_hist", f"{timestamp}_{args.exp_name}")
    experiment_config = vars(args)
    accelerator.init_trackers(tb_log_dir, experiment_config)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    total_loss = 0.  

    # Potentially load in the weights and states from a previous save
    resume_step = None
    if args.resume_from_checkpoint:
        # if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
        checkpoint_path = args.resume_from_checkpoint
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path, map_location="cpu", strict=False)
        
        # Extract `step_{i}`
        training_difference = os.path.splitext(os.path.basename(checkpoint_path))[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        lr_scheduler.step_num = completed_steps
        resume_step -= starting_epoch * len(train_dataloader) # relative step

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        optimizer.zero_grad()

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # avg loss
            loss = loss / args.gradient_accumulation_steps
            total_loss += loss.detach().item()
            accelerator.backward(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:

                # gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # update, sync gradient
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # n_steps + 1
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                    # save state
                    if accelerator.is_main_process:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir, safe_serialization=False)
                        
                        current_lr = optimizer.param_groups[0]['lr']
                        accelerator.log({"learning_rate": current_lr},
                                        step=completed_steps,)

                    # gather and compute mean loss
                    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
                    global_loss = accelerator.gather(total_loss_tensor).sum().item()
                    avg_train_loss = global_loss / accelerator.num_processes / checkpointing_steps

                    # reset total_loss = 0
                    total_loss = 0.

                    # eval model
                    model.eval()
                    losses = []
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)

                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    if accelerator.is_main_process:
                        accelerator.log(
                            {
                                "perplexity": perplexity,
                                "eval_loss": eval_loss,
                                "train_loss": avg_train_loss,
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )

            if completed_steps >= args.max_train_steps:
                break

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
