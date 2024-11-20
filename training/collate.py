import torch
import torchvision.models as models

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.add_bos_token('<bos>')
tokenizer.add_eos_token('<eos>')
tokenizer.add_pad_token('<pad>')
print(tokenizer.special_tokens_map)
print('map of special token:', tokenizer.special_tokens_map)
# Load transforms from torchvision vit_b_16
transforms = models.ViT_B_16_Weights \
                .IMAGENET1K_V1 \
                .transforms()

def collate_fn(batch):

    # Separate patches and captions
    images_list = [transforms(item['image']) for item in batch]
    images_tensor = torch.stack(images_list)

    captions = [item['caption'] for item in batch]
    print(captions)

    # Tokenize and pad captions
    captions_encoding = tokenizer(
        captions,
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=True
    )
    print(captions_encoding)
    print(tokenizer.decode(captions_encoding['input_ids'][0]))

    return {
        'images': images_tensor,
        'input_ids': captions_encoding['input_ids'],
        'attention_mask': captions_encoding['attention_mask']
    }