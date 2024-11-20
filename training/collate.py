import torch
import torchvision.models as models

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

special_tokens = {
    'bos_token': '<|startoftext|>',
    'eos_token': '<|endoftext|>',
    'pad_token': '<|pad|>'
}

tokenizer.add_special_tokens(special_tokens)

bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id

# Load transforms from torchvision vit_b_16
transforms = models.ViT_B_16_Weights \
                .IMAGENET1K_V1 \
                .transforms()

def collate_fn(batch):

    # Separate images and captions
    images_list = [transforms(item['image']) for item in batch]
    images_tensor = torch.stack(images_list)

    captions = [item['caption'] for item in batch]
    # print(captions)

    # Tokenize and pad captions
    captions_encoding = tokenizer(
        captions,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    # Add <bos>, <eos> tokens
    input_ids = captions_encoding['input_ids']

    bos = torch.tensor([bos_token_id] * input_ids.shape[0]).unsqueeze(1)
    inputs = torch.cat([bos, input_ids], dim=1)

    pad = torch.tensor([pad_token_id] * input_ids.shape[0]).unsqueeze(1)
    targets = torch.cat([input_ids, pad], dim=1)
    mask = (targets == pad_token_id)
    first_pad_idx = mask.cumsum(dim=1).eq(1) & mask
    targets[first_pad_idx] = eos_token_id


    # Extend attention mask with two additional columns
    attention_mask = captions_encoding['attention_mask']
    prepad = torch.tensor([1] * input_ids.shape[0]).reshape(input_ids.shape[0], 1)
    attention_mask = torch.cat([prepad, attention_mask], dim=1).to(torch.bool)
    # print(attention_mask, attention_mask.shape)

    return {
        'images': images_tensor,
        'captions': inputs,
        'targets': targets,
        'attention_mask': attention_mask
    }
    