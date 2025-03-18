from functools import partial
import argparse
from os import path as osp
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DefaultDataCollator, AutoModel, AutoModelForMaskedLM
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn import preprocessing
from tqdm.auto  import tqdm

WINDOW_SIZE_BP = 1536
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class DNAEmbeddingModel(nn.Module):
    """Wrapper around HF model.

    Args:
        model_name_or_path: str, path to HF model.
    """
    def __init__(
            self,
            model_name_or_path: str,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        if "MutBERT" in model_name_or_path:
            self.backbone = AutoModel.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True,
                                                      rope_scaling={'type': 'dynamic','factor': 4.0}
                                                      )
        elif "nucleotide-transformer" in model_name_or_path:
            # NT LM `backbone` is under the `.esm` attribute
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name_or_path, trust_remote_code=True).esm
        else:
            self.backbone = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    def forward(self, input_ids):
        """Backbone forward pass to retrieve last_hidden_state."""
        if "DNABERT" in self.model_name_or_path:
            return self.backbone(input_ids)[0]
        
        return self.backbone(input_ids).last_hidden_state


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seq_len", type=int, default=512,  # 2048
                        help="Sequence length (in bp)..")
    parser.add_argument("--bp_per_token", type=int, default=1,
                        help="Number of base pairs per token.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--downstream_save_dir", type=str, default="output",
                        help="Directory to save downstream task.")
    parser.add_argument("--name", type=str, default=None, help="Embeddings model name.")
    parser.add_argument("--rcps", default=False, action="store_true", help="Use RCPS.")
    parser.add_argument("--no-rcps", dest="rcps", action="store_false", help="Do not use RCPS.")
    parser.add_argument("--embed_dump_batch_size", type=int, default=1,
                        help="Batch size for embedding dump.")
    args = parser.parse_args()
    return args


def string_reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    STRING_COMPLEMENT_MAP = {
        "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
        "N": "N", "n": "n",
    }
    
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


def recast_chromosome_tissue_dist2TSS(examples):
    """Recast chromosome to int."""
    return {
        "chromosome": -1 if examples["chromosome"] == "X" else int(examples["chromosome"]),
        "tissue": examples["tissue"],
        "distance_to_nearest_tss": examples["distance_to_nearest_tss"]
    }


def tokenize_variants(examples, tokenizer, max_length: int):
    """Tokenize sequence.

    Args:
        examples: (batch of) items from the dataset.
        tokenizer: AutoTokenizer.
        max_length: int.
    Returns:
        dict with values as list of token ids.
    """

    ref_tokenized = tokenizer.batch_encode_plus(
        examples["ref_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )
    alt_tokenized = tokenizer.batch_encode_plus(
        examples["alt_forward_sequence"],
        add_special_tokens=False,
        return_attention_mask=False,
        max_length=max_length,
        truncation=True,
    )

    return {
        "ref_input_ids": ref_tokenized["input_ids"],
        "alt_input_ids": alt_tokenized["input_ids"],
        # "ref_rc_input_ids": ref_rc_tokenized["input_ids"],
        # "alt_rc_input_ids": alt_rc_tokenized["input_ids"],
    }


def find_variant_idx(examples):
    """Find token location that differs between reference and variant sequence.

    Args:
        examples: items from the dataset (not batched).
    Returns:
        dict with values index of difference.
    """
    # Guess that variant is at halfway point
    idx = len(examples["ref_input_ids"]) // 2
    if examples["ref_input_ids"][idx] == examples["alt_input_ids"][idx]:
        # If no, loop through sequence and find variant location
        idx = -1
        for i, (ref, alt) in enumerate(zip(examples["ref_input_ids"], examples["alt_input_ids"])):
            if ref != alt:
                idx = i
    # Same as above, but for reverse complement
    # rc_idx = len(examples["ref_rc_input_ids"]) // 2 - 1
    # if examples["ref_rc_input_ids"][rc_idx] == examples["alt_rc_input_ids"][rc_idx]:
    #     rc_idx = -1
    #     for i, (ref, alt) in enumerate(zip(examples["ref_rc_input_ids"], examples["alt_rc_input_ids"])):
    #         if ref != alt:
    #             rc_idx = i
    return {"variant_idx": idx}  #, "rc_variant_idx": rc_idx}


def prepare_dataset(args, tokenizer):
    """Prepare or load the tokenized dataset."""
    # Data Preprocessing
    num_tokens = args.seq_len // args.bp_per_token

    # Load data
    cache_dir = osp.join(
        "data", "variant_effect_causal_eqtl", f"seqlen={args.seq_len}"
        # "InstaDeepAI_genomics-long-range-benchmark"
    )
    if "nucleotide-transformer" in args.model_name_or_path.lower():  # NT uses 6-mers, so tokenization is different
        preprocessed_cache_file = osp.join(cache_dir, "6mer_token_preprocessed")

    elif "enformer" in args.model_name_or_path.lower():
        # Enformer tokenization requires having vocab of just `A,C,G,T,N` (in that order)
        preprocessed_cache_file = osp.join(cache_dir, "enformer_char_token_preprocessed")
    else:
        preprocessed_cache_file = osp.join(cache_dir, "char_token_preprocessed")
    print(f"Cache dir: {cache_dir}")
    print(f"Cache dir preprocessed: {preprocessed_cache_file}")

    if not os.path.exists(preprocessed_cache_file):
        os.makedirs(preprocessed_cache_file, exist_ok=True)
        dataset = load_dataset(
            "InstaDeepAI/genomics-long-range-benchmark",
            task_name="variant_effect_causal_eqtl",  # variant_effect_gene_expression
            sequence_length=args.seq_len,
            cache_dir="data",
            load_from_cache=False,
        )
        print("Dataset loaded. Cached to disk:")
        print(osp.dirname(list(dataset.cache_files.values())[0][0]["filename"]))
        try:
            del dataset["validation"]  # `validation` split is empty
        except KeyError:
            pass

        # Process data
        dataset = dataset.filter(
            lambda example: example["ref_forward_sequence"].count('N') < 0.005 * args.seq_len,
            desc="Filter N's"
        )
        dataset = dataset.map(
            recast_chromosome_tissue_dist2TSS,
            remove_columns=["chromosome", "tissue", "distance_to_nearest_tss"],
            desc="Recast chromosome"
        )
        dataset = dataset.map(
            partial(tokenize_variants, tokenizer=tokenizer, max_length=num_tokens),
            batch_size=1000,
            batched=True,
            remove_columns=["ref_forward_sequence", "alt_forward_sequence"],
            desc="Tokenize"
        )
        dataset = dataset.map(find_variant_idx, desc="Find variant idx")
        dataset.save_to_disk(preprocessed_cache_file)
    
    dataset = load_from_disk(preprocessed_cache_file)
    return dataset


def concat_storage_dict_values(storage_dict):
    """Helper method that combines lists of tensors in storage_dict into a single torch.Tensor."""
    return {key: torch.cat(storage_dict[key], dim=0) for key in storage_dict.keys()}


def dump_embeddings(args, dataset, model, device):
    """Dump embeddings to disk."""
    def extract_embeddings(item_ref, item_alt, variant_idx):
        """Extract embedding representation from last layer outputs

        Args:
            item_ref: torch.Tensor, shape (batch_size, seq_len, hidden_size) Ref embedding
            item_alt: torch.Tensor, shape (batch_size, seq_len, hidden_size) Alt embedding
            variant_idx: torch.Tensor, shape (batch_size,) Index of variant
        Returns:
            layer_metrics: dict, with values to save to disk
        """
        layer_metrics = {}

        # Compute windowed statistics
        if "enformer" in args.model_name_or_path.lower():
            window_size = WINDOW_SIZE_BP // 128  # Enformer's receptive field is 128
            # We also need to override variant_idx since Enformer model reduces to target_length of 896
            variant_idx = torch.ones_like(variant_idx) * item_ref.size(1) // 2
        else:
            window_size = WINDOW_SIZE_BP // args.bp_per_token

        # Add 1 so that window is: [window // 2 - SNP - window // 2]
        start, end = -window_size // 2, window_size // 2 + 1
        expanded_indices = torch.arange(start, end, device=item_ref.device).unsqueeze(0) + \
                           variant_idx.unsqueeze(1).to(item_ref.device)
        expanded_indices = torch.clamp(expanded_indices, 0, item_ref.size(1) - 1)  # Handle boundary conditions
        tokens_window_ref = torch.gather(
            item_ref, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        tokens_window_alt = torch.gather(
            item_alt, 1,
            expanded_indices.unsqueeze(-1).expand(-1, -1, item_ref.size(2))
        ).mean(dim=1)
        layer_metrics["concat_avg_ws"] = torch.cat([tokens_window_ref, tokens_window_alt], dim=-1)
        return layer_metrics

    embeds_path = osp.join(args.downstream_save_dir, args.name)
    os.makedirs(embeds_path, exist_ok=True)

    dataloader_params = {
        "batch_size": args.embed_dump_batch_size,
        "collate_fn": DefaultDataCollator(return_tensors="pt"),
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
        "drop_last": False  # True
    }

    # Process label_encoder = preprocessing.LabelEncoder()
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(dataset["test"]["tissue"])
    train_tissue_embed = label_encoder.transform(dataset["train"]["tissue"])
    dataset["train"] = dataset["train"].add_column("tissue_embed", train_tissue_embed)
    test_tissue_embed = label_encoder.transform(dataset["test"]["tissue"])
    dataset["test"] = dataset["test"].add_column("tissue_embed", test_tissue_embed)


    if not all([
        osp.exists(osp.join(embeds_path, f"{split_name}_embeds_combined.pt")) for split_name in dataset.keys()
    ]):
        for split_name, split in dataset.items():

            dl = DataLoader(split, **dataloader_params)

            storage_dict = {
                "concat_avg_ws": [],
                # "rc_concat_avg_ws": [],
                "chromosome": [],
                "labels": [],
                "distance_to_nearest_tss": [],
                "tissue_embed": [],
            }

            with torch.no_grad():
                for batch_idx, batch in tqdm(
                        enumerate(dl), total=len(dl), desc=f"Embedding {split_name}"
                ):
                    for key in ["chromosome", "labels", "distance_to_nearest_tss", "tissue_embed"]:
                        storage_dict[key].append(batch[key].to("cpu", non_blocking=True))
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        if "1mer" in args.model_name_or_path:
                            bs_alt_inputs = F.one_hot(batch["alt_input_ids"], num_classes=9).float().to(device)
                            bs_ref_inputs = F.one_hot(batch["ref_input_ids"], num_classes=9).float().to(device)
                        else:
                            bs_alt_inputs = batch["alt_input_ids"].to(device)
                            bs_ref_inputs = batch["ref_input_ids"].to(device)
                            # output_alt = model(batch["alt_input_ids"].to(device))
                            # output_ref = model(batch["ref_input_ids"].to(device))
                        output_alt = model(bs_alt_inputs)
                        output_ref = model(bs_ref_inputs)

                    metrics = extract_embeddings(
                        item_ref=output_ref,
                        item_alt=output_alt,
                        variant_idx=batch["variant_idx"],
                    )
                    for key, value in metrics.items():
                        storage_dict[key].append(value.to("cpu", non_blocking=True))

                storage_dict_temp = concat_storage_dict_values(storage_dict)
                torch.save(storage_dict_temp, osp.join(embeds_path, f"{split_name}_embeds.pt"))
                print(f"Saved {split_name} embeddings to {osp.join(embeds_path, f'{split_name}_embeds.pt')}")
    else:
        print("Embeddings already exist, skipping!")


def main(args):
    """Main entry point."""
    # Setup device
    device = torch.device("cuda")

    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              model_max_length=args.seq_len,
                                              trust_remote_code=True)

    # Get dataset
    dataset = prepare_dataset(args, tokenizer)

    # Get model
    model = DNAEmbeddingModel(args.model_name_or_path).to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    print("Model loaded.")

    # Dump embeddings
    dump_embeddings(args, dataset, model, device)


if __name__ == "__main__":
    args = get_args()
    main(args)

