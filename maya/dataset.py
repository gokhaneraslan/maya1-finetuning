import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .config import TrainConfig
from .constants import (
    PAD_TOKEN_ID, SOH_ID, BOS_ID, 
    EOH_ID, SOA_ID, TEXT_EOT_ID,
    CODE_START_TOKEN_ID, CODE_END_TOKEN_ID,
)


class Maya1Dataset(Dataset):

    def __init__(self, config: TrainConfig):
        self.config = config
        
        self.files = sorted(f for f in os.listdir(config.preprocessed_dir) if f.endswith(".pt"))
        
        if not self.files:
            raise RuntimeError(f"No .pt files found in '{config.preprocessed_dir}'. Run preprocess.py first.")
            
        print(f"{len(self.files)} samples loaded.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        
        path = os.path.join(self.config.preprocessed_dir, self.files[idx])
        
        try:
            
            data = torch.load(path, weights_only=True)

            text_tokens = data["text_tokens"]
            snac_tokens = data["snac_tokens"]

            prefix = torch.tensor([SOH_ID, BOS_ID], dtype=torch.long)
            middle = torch.tensor([TEXT_EOT_ID, EOH_ID, SOA_ID, CODE_START_TOKEN_ID], dtype=torch.long)
            suffix = torch.tensor([CODE_END_TOKEN_ID], dtype=torch.long)

            input_ids = torch.cat([prefix, text_tokens, middle, snac_tokens, suffix])
            labels = input_ids.clone()

            audio_start = len(prefix) + len(text_tokens) + len(middle)
            labels[:audio_start] = -100

            return {
                "input_ids": input_ids, 
                "labels": labels
            }

        except Exception as e:
            print(f"Failed to load sample {idx} ({path}): {e}")
            return None


def data_collator(batch):

    batch = [item for item in batch if item is not None]

    if not batch:
        return {}

    input_ids = pad_sequence(
        [x["input_ids"] for x in batch],
        batch_first=True,
        padding_value=PAD_TOKEN_ID,
    )
    labels = pad_sequence(
        [x["labels"] for x in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": (input_ids != PAD_TOKEN_ID).long(),
    }