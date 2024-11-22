import torch
import torch.nn as nn
from api.models.vision_transformer import CaptionModel
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from training.collate import collate_fn
from training.postgres_image_caption_dataset import PostgresImageCaptionDataset
from training.utils import calculate_metrics

model = CaptionModel()
config = {
    "batch_size": 2,
    "epochs": 3,
    "learning_rate": 0.001,
    "encoder_dim": 768,
    "decoder_dim": 768,
    "architecture": "VITencoder-GPT2decoder"
}
print(f"vocab_size: {len(model.decoder.tokenizer)}")
GPT2_VOCAB_SIZE = len(model.decoder.tokenizer)

wandb.init(project="caption_finetune", config=config)

dataset = PostgresImageCaptionDataset()
dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    best_accuracy = 0.0 
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        running_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")):
            images = batch['images'].to(device)
            captions = batch['captions'].to(device)
            targets = batch['targets'].to(device)
            attention_masks = batch['attention_mask'].to(device)

            targets = targets  
          
            optimizer.zero_grad()

            outputs = model(images, captions, attention_masks)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
            targets = targets.reshape(-1)  # Shape: (batch_size * sequence_length)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        # with torch.no_grad():
        #     val_outputs = model(images, captions, attention_masks)  # TODO Use validation data here
        #     val_outputs = val_outputs.view(-1, val_outputs.size(-1))
        #     val_targets = targets.view(-1)

        #     accuracy, precision = calculate_metrics(val_outputs, val_targets)
        wandb.log({
            "epoch": epoch + 1,
            "loss": running_loss / len(dataloader),
            })

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     torch.save(model.state_dict(), "caption_model.pth")
        #     print(f"Best model saved with accuracy: {best_accuracy:.4f}")

        print(f'Epoch [{epoch + 1}/{config["epochs"]}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

    wandb.finish()

if __name__ == "__main__":
    
    train()
