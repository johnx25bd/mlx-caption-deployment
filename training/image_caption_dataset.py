import torch
from datasets import Dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # Set tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset # Set dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # Variable-length list of patches
        caption = item['caption']
        if isinstance(caption, list):
            caption = caption[0]  # Use the first caption if multiple

        # Convert patches to a tensor
        # patches_tensor = torch.tensor(patches, dtype=torch.float32)

        # **Add this line to check the shape**
        # print(f"Sample {idx}, image: {image}")

        return {
            'image': image,
            'caption': caption
        }