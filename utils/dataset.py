"""
PyTorch Dataset for the Dates Generator task.
"""
from __future__ import annotations
import os
import random
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from utils.tokenizer import (
    parse_line, encode_conditions, date_to_tokens,
    PAD_IDX, MAX_DATE_LEN
)


class DatesDataset(Dataset):
    """
    Each sample: (day_idx, month_idx, leap_idx, decade_idx, date_token_ids)
    date_token_ids: [BOS, c0, c1, ..., c9, EOS]  length = MAX_DATE_LEN + 2 = 12
    """

    def __init__(self, filepath: str,
                 split: str = "train",
                 train_ratio: float = 0.85,
                 seed: int = 42):
        super().__init__()
        all_samples = self._load(filepath)
        random.seed(seed)
        random.shuffle(all_samples)
        n_train = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = all_samples[:n_train]
        else:
            self.samples = all_samples[n_train:]

    def _load(self, filepath: str) -> List[Tuple]:
        samples = []
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Missing training data: {filepath}\n"
                f"Generate it with:\n"
                f"  python scripts/generate_data.py"
            )
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                rec = parse_line(line)
                if rec is None or rec["date"] is None:
                    continue
                day_i, mon_i, leap_i, dec_i = encode_conditions(
                    rec["day"], rec["month"], rec["leap"], rec["decade"]
                )
                tok = date_to_tokens(rec["date"])
                samples.append((day_i, mon_i, leap_i, dec_i, tok))
        if not samples:
            raise ValueError(f"No valid dated samples found in {filepath}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        day_i, mon_i, leap_i, dec_i, tok = self.samples[idx]
        # Input tokens for teacher-forcing: BOS + date chars (no EOS)
        input_tok  = tok[:-1]   # length 11
        target_tok = tok[1:]    # length 11  (date chars + EOS)
        return (
            torch.tensor(day_i,  dtype=torch.long),
            torch.tensor(mon_i,  dtype=torch.long),
            torch.tensor(leap_i, dtype=torch.long),
            torch.tensor(dec_i,  dtype=torch.long),
            torch.tensor(input_tok,  dtype=torch.long),
            torch.tensor(target_tok, dtype=torch.long),
        )


class ConditionsDataset(Dataset):
    """For inference: only conditions, no date."""

    def __init__(self, filepath: str):
        self.samples = []
        self.raw_conditions = []
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Missing input conditions: {filepath}\n"
                f"Generate example inputs with:\n"
                f"  python scripts/generate_data.py"
            )
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                rec = parse_line(line)
                if rec is None:
                    continue
                day_i, mon_i, leap_i, dec_i = encode_conditions(
                    rec["day"], rec["month"], rec["leap"], rec["decade"]
                )
                self.samples.append((day_i, mon_i, leap_i, dec_i))
                self.raw_conditions.append(
                    (rec["day"], rec["month"], rec["leap"], rec["decade"])
                )
        if not self.samples:
            raise ValueError(f"No valid condition rows found in {filepath}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        day_i, mon_i, leap_i, dec_i = self.samples[idx]
        return (
            torch.tensor(day_i,  dtype=torch.long),
            torch.tensor(mon_i,  dtype=torch.long),
            torch.tensor(leap_i, dtype=torch.long),
            torch.tensor(dec_i,  dtype=torch.long),
        )
