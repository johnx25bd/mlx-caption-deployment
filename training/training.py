import torch
import torch.nn as nn
from api.models.vision_transformer import CaptionModel
from torch.utils.data import DataLoader #, Data
import wandb
from sklearn.metrics import precision_score as sk_precision_score
from tqdm import tqdm
# from datafile import data  
from training.image_caption_dataset import ImageCaptionDataset
from training.collate import collate_fn
from datasets import load_from_disk, load_dataset



# Hyperparameters
config = {
    "batch_size": 2,
    "epochs": 3,
    "learning_rate": 0.001,
    "encoder_dim": 768,  # Hardcoded (?) dimension
    "decoder_dim": 768,  # Hardcoded (?) dimension
    "architecture": "VITencoder-GPT2decoder"
}

GPT2_VOCAB_SIZE = 50260

# Initialize wandb
wandb.init(project="caption_test", config=config)

# flickr30k dataset download (ignore if you already have ds on local)
# ds = load_dataset("nlphuji/flickr30k", split="test") 

# Optionally subset the dataset
# ds = ds.select(range(NUM_OF_IMAGES)) # Select the first X images w/ their captions

# ds.save_to_disk("flickr_ds")

try:
    ds = load_from_disk("./flickr_ds") # patched images, untokenized captions
except: 
    ds = load_from_disk("./training/flickr_ds") # patched images, untokenized captions

dataset = ImageCaptionDataset(ds)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=False)

batch = next(iter(dataloader))
# print(batch['images'].shape)
# print(batch['input_ids'].shape)
# print(batch['attention_mask'].shape)


def calculate_metrics(outputs, targets):
    _, predicted = torch.max(outputs, 1)  # Get the predicted class
    accuracy = (predicted == targets).float().mean().item()  # Calculate accuracy
    precision = sk_precision_score(targets, predicted, average='weighted', zero_division=0)  # Calculate precision
    return accuracy, precision

# Training loop
best_accuracy = 0.0 

def train():

    best_accuracy = 0.0 
# Initialize model
    model = CaptionModel()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        running_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")):

            images = batch['images']
            captions = batch['captions']
            targets = batch['targets']
            attention_masks = batch['attention_mask']

            encoder_input = images  # Batch of images for the encoder
            decoder_input = captions #[:, :-1]  # All but the last token for decoder input
            targets = targets  
          
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(encoder_input, decoder_input, attention_masks)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
            targets = targets.reshape(-1)  # Shape: (batch_size * sequence_length)

            # Calculate loss
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(encoder_input, decoder_input, attention_masks)  # Use validation data here
            val_outputs = val_outputs.view(-1, val_outputs.size(-1))
            val_targets = targets.view(-1)

            accuracy, precision = calculate_metrics(val_outputs, val_targets)
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": running_loss / len(dataloader),
            # "accuracy": accuracy,
            # "precision": precision
            })

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "caption_model.pth")  # Save the best model
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

    # Finish wandb run
    wandb.finish()

# print('no error yet 2')

if __name__ == "__main__":
    
    train()
    # wandb.finish()
# wandb.finish()
