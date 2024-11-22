import torch
import torch.nn as nn
from api.models.vision_transformer import CaptionModel
from torch.utils.data import DataLoader #, Data
import wandb
from tqdm import tqdm
# from datafile import data
from training.image_caption_dataset import ImageCaptionDataset
from training.collate import collate_fn
from datasets import load_from_disk, load_dataset

from training.utils import calculate_metrics



# Hyperparameters
config = {
    "batch_size": 30,
    "epochs": 10,
    "learning_rate": 0.001,
    "encoder_dim": 768,  # Hardcoded (?) dimension
    "decoder_dim": 768,  # Hardcoded (?) dimension
    "architecture": "VITencoder-GPT2decoder"
}

GPT2_VOCAB_SIZE = 50260

# Initialize wandb
wandb.init(project="caption_test", config=config)

# flickr30k dataset download (ignore if you already have ds on local)
ds = load_dataset("nlphuji/flickr30k", split="test") 

# Optionally subset the dataset
ds = ds.select(range(10000)) # Select the first X images w/ their captions

# ds.save_to_disk("flickr_ds")

# try:
#     ds = load_from_disk("./flickr_ds") # patched images, untokenized captions
# except: 
#     ds = load_from_disk("./training/flickr_ds") # patched images, untokenized captions

dataset = ImageCaptionDataset(ds)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch = next(iter(dataloader))

def train():
    best_accuracy = 0.0 
# Initialize model
    model = CaptionModel()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print_interval = 10

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        running_loss = 0.0
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")):
            images = batch['images'].to(device)
            captions = batch['captions'].to(device)
            targets = batch['targets'].to(device)
            attention_masks = batch['attention_mask'].to(device)

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
            epoch_loss += loss.item()

            # Log and print metrics every `print_interval` batches
            if (i + 1) % print_interval == 0:
                avg_loss = running_loss / print_interval

                # Compute metrics (on current batch for simplicity)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == targets).float().mean().item()
                precision = sk_precision_score(
                    targets.cpu(),
                    predicted.cpu(),
                    average='weighted',
                    zero_division=0
                )

                # Log to wandb
                wandb.log({
                    "batch_loss": avg_loss,
                    "batch_accuracy": accuracy,
                    "batch_precision": precision,
                    "step": i + epoch * len(dataloader)
                })

                # Print metrics
                print(f"Epoch [{epoch + 1}/{config['epochs']}], Step [{i + 1}/{len(dataloader)}], "
                      f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

                running_loss = 0.0  # Reset running loss for the next interval



        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_outputs = model(encoder_input, decoder_input, attention_masks)  # Use validation data here
            val_outputs = val_outputs.view(-1, val_outputs.size(-1))
            val_targets = targets.view(-1)
            accuracy, precision = calculate_metrics(val_outputs, val_targets)
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss / len(dataloader),
            "accuracy": accuracy,
            "precision": precision
            })

        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "caption_model.pth")  # Save the best model
            print(f"Best model saved with accuracy: {best_accuracy:.4f}")

        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

    wandb.finish()


if __name__ == "__main__":
    train()
