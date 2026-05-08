"""Utility functions for logging, checkpointing, and profiling."""


def setup_logging(config, rank=0):
    """Initialize logging (wandb, tensorboard, console)."""
    if rank != 0:
        return
    wandb_project = config.get("logging", {}).get("wandb_project")
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project)
        except ImportError:
            print("wandb not installed, skipping")
    print(f"Logging initialized for project: {wandb_project}")