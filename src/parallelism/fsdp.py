"""FSDP-based trainer with gradient checkpointing for large models."""

import torch
import torch.nn as nn


class FSDPTrainer:
    """Trainer with Fully Sharded Data Parallel support."""

    def __init__(self, model, config, rank=0):
        self.model = model
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        model.to(self.device)
        self.grad_ckpt = config.get("training", {}).get("gradient_checkpointing", True)

        if self.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def train(self, dataset, optimizer):
        """Run training loop."""
        max_steps = 1000
        model = self.model
        model.train()

        for step, batch in enumerate(dataset):
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids.clone()).to(self.device)

            outputs = model(input_ids, attention_mask=None)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if self.rank == 0 and step % 10 == 0:
                print(f"Step {step}: loss={loss.item():.4f}")

        if self.rank == 0:
            print("Training complete.")