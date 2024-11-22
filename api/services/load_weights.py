import torch
import wandb

def load_weights(device):
    wandb.login(key="fce0295fbd7f54f738cd69fbeba699b57c81e93f")
    run = wandb.init()

    # Fetch and download the artifact
    artifact = run.use_artifact('machine-learning-institute/caption_model/caption_model:v0', type='model')
    artifact_dir = artifact.download()  # Returns the path to the directory where artifact is saved

    # Path to the state_dict file
    state_dict_path = f"{artifact_dir}/caption_model.pth"
    state_dict = torch.load(state_dict_path, map_location=torch.device(device), weights_only=True) 
    return state_dict
