"""PyTorch Dataset wrapper around tokenized reviews."""
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)
