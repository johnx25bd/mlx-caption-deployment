from torch.utils.data import DataLoader
from image_caption_dataset import ImageCaptionDataset
from collate import collate_fn
from datasets import load_from_disk

ds = load_from_disk("./patched_ds") # patched images, untokenized captions

dataset = ImageCaptionDataset(ds)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

batch = next(iter(dataloader))
print(batch['images'].shape)
print(batch['input_ids'].shape)
print(batch['attention_mask'].shape)

# 
# 
# for batch in dataloader:
#     images = batch['images'] # [batch_size, max_num_patches, patch_dim, patch_dim, channels]
#     flattened_patches = patches.reshape(patches.shape[0], patches.shape[1], -1) # Flatten last three dimensions
#     captions = batch['input_ids']
#     # Optionally, you can get the attention masks if needed
#     attention_mask = batch['attention_mask']
#     print("Flattened patches shape:", flattened_patches.shape)  # Should be [batch_size, max_num_patches, patch_dim * patch_dim * channels]
#     print("Captions shape:", captions.shape)  # Should be [batch_size, max_seq_length]
#     break