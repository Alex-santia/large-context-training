"""Optimizer and learning rate scheduler factory."""

import torch


def create_optimizer(model, training_config):
    """Create optimizer with configurable LR scheduling."""
    lr = training_config.get("learning_rate", 2e-5)
    weight_decay = training_config.get("weight_decay", 0.1)
    warmup_steps = training_config.get("warmup_steps", 200)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer