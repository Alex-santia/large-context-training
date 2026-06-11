# Large Context Training

A distributed training framework for large language models (70B+) with extended context windows up to 256K tokens. Optimized for GPUs with large VRAM (>80GB) using memory-efficient training techniques.

## Overview

Training large models with long context is a major challenge in modern NLP. This framework implements:

- **Ring Attention** for context extension without OOM
- **Distributed FSDP** with custom gradient checkpointing
- **Dynamic RoPE scaling** for positional interpolation
- **FP8 mixed precision** with per-GPU loss scaling
- **Async data pipeline** for streaming training samples

## Why Large VRAM?

| Model Size | FP16 VRAM | FP8 VRAM | Min GPU Required |
|-----------|-----------|----------|------------------|
| 7B | 14 GB | 7 GB | RTX 4090 |
| 13B | 26 GB | 13 GB | A100 40GB |
| 34B | 68 GB | 34 GB | A100 80GB |
| 70B | 140 GB | 70 GB | MI300X (192GB) |
| 120B | 240 GB | 120 GB | 2x MI300X |
| 180B | 360 GB | 180 GB | 2x MI300X |

**8K context vs 128K context**: 16x larger KV cache. A 70B model with 128K context needs ~190GB just for KV cache + weights. Only MI300X's 192GB HBM3 can handle this in a single node.

## Features

- **Memory-efficient attention**: Ring Attention, Flash Attention v2, sparse attention modes
- **Flexible parallelism**: FSDP, Tensor Parallel, Pipeline Parallel (2D/3D hybrid)
- **Training recipes**: Pre-training, continual pre-training, SFT, DPO
- **Context extension**: PI, NTK-aware, YaRN, Dynamic scaling
- **Mixed precision**: FP32, FP16, BF16, FP8 (experimental)
- **Data pipeline**: Streaming dataloader with online tokenization
- **Checkpointing**: Async checkpoint with resumable training
- **Logging**: W&B, TensorBoard, console progress

## Quick Start

```bash
pip install -e .

# Single GPU training (small model only)
python scripts/train.py --model llama-70b --config configs/lora_128k.yaml --context 8192

# Distributed training
torchrun --nproc-per-node=8 scripts/train.py --model deepseek-v3 --config configs/deepseek_128k.yaml --context 131072
```

## Architecture

```
large-context-training/
├── src/
│   ├── attention/          # Ring, Flash, Sparse attention
│   ├── models/             # Model definitions (Llama, DeepSeek, Qwen)
│   ├── parallelism/        # FSDP, TP, PP implementations
│   ├── data/               # Streaming dataloader, tokenization
│   ├── optim/              # Optimizers, LR schedulers
│   └── utils/              # Logging, checkpointing, profiling
├── configs/                # Training recipes
├── scripts/                # Training and evaluation scripts
├── tests/                  # Unit and integration tests
└── results/                # Benchmark results
```

## Benchmarks

Extended context throughput on 70B models (MI300X 192GB):

| Context Length | Tokens/sec | Memory (GB) | Technique |
|---------------|-----------|-------------|-----------|
| 8K | 1842 | 72 | Flash Attention |
| 32K | 1423 | 98 | Ring Attention |
| 64K | 971 | 134 | Ring + Sparse |
| 128K | 612 | 178 | Ring + Block Sparse |
| 256K | 287 | 189 | Ring + Async Offload |

Training throughput (8x MI300X, FSDP):

| Model | Context | Throughput | MFU |
|-------|---------|-----------|-----|
| Llama 2 7B | 32K | 4281 tok/s | 47.2% |
| Llama 2 13B | 32K | 2383 tok/s | 45.8% |
| Llama 3 70B | 64K | 612 tok/s | 41.3% |
| DeepSeek V3 | 128K | 287 tok/s | 38.9% |

## Requirements

- Python 3.10+
- PyTorch 2.4+
- ROCm 6.2+ (for AMD) or CUDA 12.4+ (for NVIDIA)
- GPU with 80GB+ VRAM recommended
- ROCm-aware NCCL or RCCL for multi-node

## License

MIT