import wandb


def initialize_wandb(project_name, run_name):
    """Initialize WandB for experiment tracking."""
    wandb.init(project=project_name, name=run_name)
