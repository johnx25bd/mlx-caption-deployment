import wandb

artifact = wandb.Artifact('model_name', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)