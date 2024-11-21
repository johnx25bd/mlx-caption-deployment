import PIL
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2Tokenizer, GPT2Model

class EncoderViT(nn.Module):
    """
    Vision Transformer Encoder using torchvision's vit_b_16 model.
    Extracts the last hidden layer before the classification head.
    """
    def __init__(self):
        super().__init__()
        # Load the ViT model and weights
        self.weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = models.vit_b_16(weights=self.weights)
        self.transforms = self.weights.transforms()
        self.hidden_output = None

        # Register a forward hook to capture the last hidden layer
        self._register_hooks()

    def _register_hooks(self):
        self.vit.encoder.layers[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # Capture the hidden output, skipping the [CLS] token
        self.hidden_output = output[:, 1:, :]  # [batch_size, num_patches, hidden_dim]

    def forward(self, x):
        _ = self.vit(x)  # Trigger the forward hook
        if self.hidden_output is None:
            raise RuntimeError("Hook did not capture encoder features; ensure the forward pass runs correctly.")
        return self.hidden_output


class GPT2Decoder(nn.Module):
    """
    Decoder using GPT-2.
    Includes cross-attention to combine image features with textual inputs.
    """
    def __init__(self, gpt2_name="gpt2"):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        special_tokens = {
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>',
            'pad_token': '<|pad|>'
        }
        self.tokenizer.add_special_tokens(special_tokens)

        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        
        # Resize token embeddings to account for new special tokens
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

        # Project image features to match GPT-2's embedding size
        self.image_projection = nn.Linear(768, self.gpt2.config.hidden_size)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=8, batch_first=True)
            # TODO: Implement a ModuleList with several cross attention blocks.
        
        # Feed forward
        self.ffw = nn.Sequential(
            nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.gpt2.config.hidden_size * 4, self.gpt2.config.hidden_size),
        )
        self.logits = nn.Linear(self.gpt2.config.hidden_size, 50260) 
        # TODO: CHEEKY. Don't hardcode the vocab size.

    def forward(self, encoded_images, input_ids, attention_mask=None):
        """
        Args:
            encoded_images: Output from the ViT encoder [batch_size, num_patches, hidden_dim]
            input_ids: Input token IDs for GPT-2 [batch_size, seq_len]
            attention_mask: Attention mask for GPT-2 inputs [batch_size, seq_len]
        Returns:
            logits: The logits from GPT-2 after integrating image features
        """
        seq_len = input_ids.shape[1]
        # Pass textual inputs through GPT-2 embeddings
        embeddings = self.gpt2.wte(input_ids)  # [batch_size, seq_len, hidden_size]
        # position_encodings = self.gpt2.wpe(torch.arange(seq_len, device=input_ids.device))
        # position_encodings = position_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        extended_attention_mask = attention_mask[:, None, None, :]
        hidden_states = embeddings # + position_encodings
        for i, block in enumerate(self.gpt2.h[0:8]):
            # hidden_states = hidden_states.transpose(-3, -2) # [seq_len, batch_size, hidden_size]
            block_output = block(hidden_states, attention_mask=extended_attention_mask)[0] # do we need attention mask?
            hidden_states = block_output
        
        caption_encoding = hidden_states


        # Project image features to GPT-2's hidden size
        projected_images = self.image_projection(encoded_images)  # [batch_size, num_patches, hidden_size]

        # Apply cross-attention between image features and GPT-2 embeddings
        # TODO: Implement a ModuleList with several cross attention blocks.
        attended_output, _ = self.cross_attention(caption_encoding, projected_images, projected_images)

        # FEED FORWARD
        attended_output = self.ffw(attended_output)
        # Residual connection
        attended_output = attended_output + caption_encoding

        # OUTPUT size: (seq_len, vocab_size)
        logits = self.logits(attended_output)

        return logits


class CaptionModel(nn.Module):
    """
    Full encoder-decoder model with cross-attention.
    Combines ViT and GPT-2 for image-to-text tasks.
    """
    def __init__(self, gpt2_name="gpt2"):
        super().__init__()
        self.encoder = EncoderViT()
        self.decoder = GPT2Decoder(gpt2_name)

    def forward(self, images, input_ids, attention_mask=None):
        # Encode images
        encoded_images = self.encoder(images)  # [batch_size, num_patches, hidden_dim]

        # Decode text with cross-attention to image features
        decoded_output = self.decoder(encoded_images, input_ids, attention_mask)
        return decoded_output


if __name__ == "__main__":
    # Dummy data
    batch_size = 3
    image_size = (3, 224, 224)
    seq_len = 12

    # Random images and captions
    image = PIL.Image.open("./dog.png").convert("RGB")
    
    dummy_input_ids = torch.randint(0, 50257, (batch_size, seq_len))  # Random token IDs for GPT-2
    dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype = bool)  # Attention mask

    # Initialize the full model
    model = CaptionModel()
    tsf_image = model.encoder.transforms(image).unsqueeze(0)
    tsf_image = tsf_image.repeat(batch_size, 1, 1, 1) 

    # Preprocess images using the encoder's transform

    # Forward pass
    output = model(tsf_image, dummy_input_ids, dummy_attention_mask)
    print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, hidden_size]
