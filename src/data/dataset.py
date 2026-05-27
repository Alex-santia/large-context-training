"""Streaming dataset with online tokenization for large-scale training."""

import json
import random
from typing import Optional

import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    """
    Streaming dataset that loads and tokenizes data on-the-fly.
    Supports sharded loading for distributed training.
    """

    def __init__(
        self,
        config: dict,
        max_seq_len: int = 131072,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size
        self.dataset_name = config.get("dataset", "long-context-mix")
        self.streaming = config.get("streaming", True)
        self.packing = config.get("packing", True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rank = self.rank
        if worker_info is not None:
            rank += worker_info.id

        while True:
            # Simulate streaming data from remote source
            length = random.randint(self.max_seq_len // 2, self.max_seq_len)
            tokens = torch.randint(0, 32000, (length,), dtype=torch.long)
            yield {"input_ids": tokens, "labels": tokens.clone()}

    def __len__(self):
        return 1000000  # Approximate length for sampler