import wandb

def initialize_wandb(project_name):
    wandb.init(project=project_name)

def log_metrics(metrics_dict):
    wandb.log(metrics_dict)
