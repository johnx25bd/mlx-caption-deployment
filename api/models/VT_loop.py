import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2Tokenizer, GPT2Model
import PIL


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
        # if self.hidden_output is None:
        #     raise RuntimeError("Hook did not capture encoder features; ensure the forward pass runs correctly.")
        return self.hidden_output


class GPT2Decoder(nn.Module):
    """
    Decoder using GPT-2.
    Combines embeddings and masked self-attention from GPT-2 with cross-attention
    to integrate image features (from ViT encoder).
    """
    def __init__(self, gpt2_name="gpt2"):
        super().__init__()
        # Load the GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        
        # Project image features to match GPT-2's hidden size
        self.image_projection = nn.Linear(768, self.gpt2.config.hidden_size)

        # Cross-attention mechanism
        self.cross_attention =nn.ModuleList(nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=4, batch_first=True) for x in range(4))
        
        # Feed-forward layer
        self.ffw = nn.Sequential(
            nn.Linear(self.gpt2.config.hidden_size, self.gpt2.config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.gpt2.config.hidden_size * 4, self.gpt2.config.hidden_size),
        )
        
        # Output layer for vocabulary prediction
        self.logits = nn.Linear(self.gpt2.config.hidden_size, self.tokenizer.vocab_size)

    def forward(self, encoded_images, input_ids, attention_mask=None):

        batch_size, seq_len = input_ids.size()
        
        # Step 1: GPT-2 embeddings + masked self-attention
        # Get token embeddings and positional encodings
        token_embeddings = self.gpt2.wte(input_ids)  # [batch_size, seq_len, hidden_size]
        position_encodings = self.gpt2.wpe(torch.arange(seq_len, device=input_ids.device)).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, hidden_size]

        embeddings = token_embeddings + position_encodings
        extended_attention_mask = attention_mask[:, None, None, :]
        encoded = self.image_projection(encoded_images) 
        
        # Pass through GPT-2's transformer blocks
        for i, block in enumerate(self.gpt2.h[:4]):
            decoded = block(embeddings, attention_mask=extended_attention_mask)[0]
        # Cross-attention: Query = captions, Key/Value = image features
            cross_attended, _ = self.cross_attention(query=decoded, key=encoded, value=encoded)
        # Step 3: Feed-forward layer and residual connection
            x = self.ffw(cross_attended) + decoded   # Residual connection
        # Step 4: Project to vocabulary logits
        logits = self.logits(x)  # [batch_size, seq_len, vocab_size]

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
    image = PIL.Image.open("./tabby.jpg").convert("RGB")
    
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
