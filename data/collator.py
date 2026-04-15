"""
SubPOPCollator — batches variable-length examples from SubPOPDataset.

input_ids and attention_mask are right-padded to the longest sequence in
the batch.  option_token_ids, target_distributions, and ordinals are kept
as Python lists because they have variable lengths (different questions have
different numbers of answer choices) and cannot be naively stacked.
"""

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class SubPOPCollator:
    pad_token_id: int

    def __call__(self, batch: list) -> dict:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

        # Keep as lists — the loss function iterates per-example
        option_token_ids = [item["option_token_ids"] for item in batch]
        target_distributions = [item["target_distribution"] for item in batch]
        ordinals = [item["ordinal"] for item in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "option_token_ids": option_token_ids,
            "target_distributions": target_distributions,
            "ordinals": ordinals,
        }
