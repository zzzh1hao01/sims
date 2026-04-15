"""
SubPOPDataset — wraps the jjssuh/subpop HuggingFace dataset.

Each item is one (demographic_group, question, answer_distribution) example.
The __getitem__ returns everything the trainer needs:
  - input_ids / attention_mask  : tokenized prompt
  - option_token_ids            : vocab indices for A, B, C, ...
  - target_distribution         : human response distribution (sums to 1)
  - ordinal                     : ordinal scale positions (for Wasserstein eval)

Train/val split is done at the question (qkey) level, not the row level,
so all demographic groups for a given question stay in the same partition.
This prevents the model from seeing a question's text during training even
when its demographic group is held out.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from data.prompt_formatter import PromptFormatter


class SubPOPDataset(Dataset):
    def __init__(
        self,
        hf_split: str,               # "train" or "test"
        tokenizer,
        formatter: PromptFormatter,
        max_length: int = 512,
        val_fraction: float = 0.05,
        val_seed: int = 42,
        is_val: bool = False,
    ):
        """
        hf_split: "train" → SubPOP-Train (Pew, 3229 questions)
                  "test"  → SubPOP-Eval  (GSS,  133 questions)
        is_val:   only meaningful when hf_split="train"; carves out a
                  question-level validation subset.
        """
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.max_length = max_length

        raw = load_dataset("jjssuh/subpop", split=hf_split)

        if hf_split == "train":
            raw = self._question_level_split(raw, val_fraction, val_seed, is_val)

        self.records = list(raw)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]

        prompt = self.formatter.format(row)
        option_token_ids = self.formatter.get_option_token_ids(row["options"])

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)        # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)

        target = torch.tensor(row["responses"], dtype=torch.float32)
        ordinal = torch.tensor(row["ordinal"], dtype=torch.float32)
        opt_ids = torch.tensor(option_token_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "option_token_ids": opt_ids,
            "target_distribution": target,
            "ordinal": ordinal,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _question_level_split(dataset, val_fraction, seed, is_val):
        """
        Split unique qkeys into train/val partitions, then filter rows.
        All demographic groups for the same qkey go to the same partition.
        """
        import random

        qkeys = list({row["qkey"] for row in dataset})
        rng = random.Random(seed)
        rng.shuffle(qkeys)

        n_val = max(1, int(len(qkeys) * val_fraction))
        val_keys = set(qkeys[:n_val])
        train_keys = set(qkeys[n_val:])

        keep_keys = val_keys if is_val else train_keys
        return dataset.filter(lambda row: row["qkey"] in keep_keys)
