#!/usr/bin/env python3
"""Training script for large context language models."""

import argparse
import os

import torch
import torch.distributed as dist

from src.models.modeling import create_model
from src.data.dataset import StreamingDataset
from src.parallelism.fsdp import FSDPTrainer
from src.optim.scheduler import create_optimizer
from src.attention.ring import RingAttention


def parse_args():
    parser = argparse.ArgumentParser(description="Train large context LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--model", type=str, default="llama-70b", help="Model name")
    parser.add_argument("--context", type=int, default=8192, help="Context length")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = 0

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)

    if rank == 0:
        print(f"=== Large Context Training ===")
        print(f"Model: {args.model}")
        print(f"Context: {args.context}")
        print(f"Config: {args.config}")

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = create_model(
        name=args.model,
        dtype=torch.bfloat16,
        rope_scaling=config.get("model", {}).get("rope_scaling"),
    )

    trainer = FSDPTrainer(
        model=model,
        config=config,
        rank=rank,
    )

    dataset = StreamingDataset(
        config=config["data"],
        max_seq_len=args.context,
    )

    optimizer = create_optimizer(model, config["training"])
    trainer.train(dataset, optimizer)

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()