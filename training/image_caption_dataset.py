import torch
from datasets import Dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

class ImageCaptionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # Variable-length list of patches
        caption = item['caption']
        if isinstance(caption, list):
            caption = caption[0]  # Use the first caption if multiple

        # **Add this line to check the shape**
        # print(f"Sample {idx}, image: {image}")

        return {
            'image': image,
            'caption': caption
        }