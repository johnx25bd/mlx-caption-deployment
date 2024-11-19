from torch.utils.data import DataLoader
from image_caption_dataset import ImageCaptionDataset
from collate import collate_fn
from datasets import load_from_disk

ds = load_from_disk("patched_ds") # patched images, untokenized captions

dataset = ImageCaptionDataset(ds)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)