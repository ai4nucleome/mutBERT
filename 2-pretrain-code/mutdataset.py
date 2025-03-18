import numpy as np
import torch
from torch.utils.data import Dataset
import os


class DNADatasetOH(Dataset):
    def __init__(self, data_path, seq_length=510, vocab_size=9, split="train", total=10, temperature=1.0):
        """
        :param data_path: path, The one-hot encoded data of shape (L, V)
        :param seq_length: The length of the segment to extract
        :param vocab_size: The size of the vocabulary (V, )
        """
        if split == "test":
            data_shape = (39158887, vocab_size)
            start_shape = (39135892,)

        elif split == "train":
            data_shape = (2872048919, vocab_size)  
            start_shape = (2871683043,)
        
        self.total = total  # 30000000 or 150000
        self.data = np.memmap(os.path.join(data_path, f"{split}_data.npy"), dtype=np.float32,
                              mode='r', shape=data_shape)
        self.all_start_list = np.memmap(os.path.join(data_path, f"{split}_start_indices.npy"),
                                        mode="r", dtype=np.int64,
                                        shape=start_shape)

        self.start_list = np.random.choice(self.all_start_list, replace=True, size=self.total)

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.temperature = temperature

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        length = self.seq_length + 2
        input_ids = torch.zeros((length, self.vocab_size), dtype=torch.float32)
        input_ids[0, 1] = 1   # [CLS]: 1
        input_ids[-1, 2] = 1  # [SEP]: 2

        start_idx = self.start_list[idx]
        memmap_sample = self.data[start_idx: start_idx + self.seq_length]
        sample = torch.from_numpy(memmap_sample.astype(np.float32))
        
        if self.temperature != 1.0:
            # temp scaling
            mask = sample != 0
            masked_sample = torch.where(mask, sample, torch.tensor(float('-inf')))
            sample = torch.nn.functional.softmax(masked_sample / self.temperature, dim=-1)

        input_ids[1:-1] = sample 

        attention_mask = torch.ones(length, dtype=torch.int64)
        tokenized_sample = {"input_ids": input_ids,
                            "attention_mask": attention_mask}
        return tokenized_sample
    