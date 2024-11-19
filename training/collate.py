import torch
from transformers import GPT2Tokenizer

def collate_fn(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Separate patches and captions
    patches_list = [item['patches'] for item in batch]
    captions = [item['caption'] for item in batch]

    # Determine the maximum number of patches in the batch
    max_num_patches = max(patches.size(0) for patches in patches_list)
    # Get the patch dimensions (e.g., [16, 16, 3])
    patch_dims = patches_list[0].size()[1:]  # Assuming all patches have the same size

    # Pad patches
    padded_patches = []
    for patches in patches_list:
        num_patches = patches.size(0)
        padding_size = max_num_patches - num_patches
        if padding_size > 0:
            # Create padding tensor with appropriate shape
            padding_shape = (padding_size,) + patch_dims  # e.g., (padding_size, 16, 16, 3)
            padding = torch.zeros(padding_shape, dtype=torch.float32)
            patches = torch.cat([patches, padding], dim=0)
        padded_patches.append(patches)
    patches_tensor = torch.stack(padded_patches)  # Shape: [batch_size, max_num_patches, 16, 16, 3]

    # Tokenize and pad captions
    captions_encoding = tokenizer(
        captions,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    return {
        'patches': patches_tensor,
        'input_ids': captions_encoding['input_ids'],
        'attention_mask': captions_encoding['attention_mask']
    }