import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import os
import json
import argparse
from datasets import load_dataset


load_dataset('flickr30k')

parser = argparse.ArgumentParser()
parser.add_argument('--nThreads', type=int, default=4)
args = parser.parse_args()

imagenet_data = torchvision.datasets.Flickr30k('./flickr30k_images', ann_file='./flickr30k_images/dataset_flickr30k.json')
dataloader = torch.utils.data.DataLoader(imagenet_data,
                                         batch_size=4,
                                         shuffled=False,
                                         num_workers=args.nThreads)

print(dataloader)