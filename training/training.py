import torch
import torch.nn as nn
from models.vision_transformer import CaptionModel
from torch.utils import Data, Dataloader
import wandb
from sklearn.metrics import precision_score
from tqdm import tqdm
# from datafile import data  



# # Hyperparameters
# config = {
#     "batch_size": 32,
#     "epochs": 10,
#     "learning_rate": 0.001,
#     "encoder_dim": 768,  # Example dimension
#     "decoder_dim": 768,  # Example dimension
#     "architecture": "VITencoder-GPT2decoder"
# }

# # Initialize wandb
# wandb.init(project="caption_test", config=config)


# # data = Flicker() # assumes I receive PIL images and tokenized captions 
# # dataloader = torch.utils.data.DataLoader(data, batch_size=config['batch_size'], collate_fn = Flicker().collate_fn, shuffle=True)

# def calculate_metrics(outputs, targets):
#     _, predicted = torch.max(outputs, 1)  # Get the predicted class
#     accuracy = (predicted == targets).float().mean().item()  # Calculate accuracy
#     precision = precision_score(targets, predicted, average='weighted', zero_division=0)  # Calculate precision
#     return accuracy, precision

# # Training loop
# best_accuracy = 0.0 

# def train():

#     best_accuracy = 0.0 
# # Initialize model
#     model = CaptionModel(
#         patch_size=16,
#         encoder_dim=config['encoder_dim'],
#         vocab_size=50257,
#         decoder_dim=config['decoder_dim'],
#         encoder_layers=config['encoder_layers'],
#         decoder_layers=config['decoder_layers'],
#     )
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    
#     for epoch in range(config["epochs"]):
#         running_loss = 0.0
#         for i, (image, caption) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")):
            
#             caption = torch.tensor(caption, dtype=torch.long)
#             encoder_input = image  # Patches for the encoder
#             decoder_input = caption[:, :-1]  # All but the last token for decoder input
#             targets = caption[:, 1:]  # All but the first token for loss target

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(encoder_input, decoder_input, )

#             # Reshape outputs and targets for loss calculation
#             outputs = outputs.view(-1, outputs.size(-1))  # Shape: (batch_size * sequence_length, vocab_size)
#             targets = targets.reshape(-1)  # Shape: (batch_size * sequence_length)

#             # Calculate loss
#             loss = criterion(outputs, targets)
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
        
#         model.eval()  # Set model to evaluation mode
#         with torch.no_grad():
#             val_outputs = model(encoder_input, decoder_input)  # Use validation data here
#             val_outputs = val_outputs.view(-1, val_outputs.size(-1))
#             val_targets = targets.view(-1)

#             accuracy, precision = calculate_metrics(val_outputs, val_targets)
#             # Log metrics to wandb
#         wandb.log({
#             "epoch": epoch + 1,
#             "loss": running_loss / len(dataloader),
#             "accuracy": accuracy,
#             "precision": precision
#                     })

#         if (i + 1) % 10 == 0:  # Print every 10 batches
#             print(f'Epoch [{epoch + 1}/{config['epochs']}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             torch.save(model.state_dict(), "mnist_model.pth")  # Save the best model
#             print(f"Best model saved with accuracy: {best_accuracy:.4f}")

#         print(f'Epoch [{epoch + 1}/{config['epochs']}], Loss: {running_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

#     # Finish wandb r
# print('no error yet 2')

# if __name__ == "__main__":
    
#     train()
#     wandb.finish()