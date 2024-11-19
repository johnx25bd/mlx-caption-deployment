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
        # Load the GPT-2 model
        self.gpt2 = GPT2Model.from_pretrained(gpt2_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        
        # Project image features to match GPT-2's embedding size
        self.image_projection = nn.Linear(768, self.gpt2.config.hidden_size)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.gpt2.config.hidden_size, num_heads=8)

    def forward(self, encoded_images, input_ids, attention_mask=None):
        """
        Args:
            encoded_images: Output from the ViT encoder [batch_size, num_patches, hidden_dim]
            input_ids: Input token IDs for GPT-2 [batch_size, seq_len]
            attention_mask: Attention mask for GPT-2 inputs [batch_size, seq_len]
        Returns:
            logits: The logits from GPT-2 after integrating image features
        """
        # Pass textual inputs through GPT-2 embeddings
        embeddings = self.gpt2.wte(input_ids)  # [batch_size, seq_len, hidden_size]

        # Project image features to GPT-2's hidden size
        projected_images = self.image_projection(encoded_images)  # [batch_size, num_patches, hidden_size]

        # Apply cross-attention between image features and GPT-2 embeddings
        # Transpose for compatibility with MultiheadAttention: [seq_len, batch_size, hidden_size]
        embeddings = embeddings.transpose(0, 1)
        projected_images = projected_images.transpose(0, 1)
        attended_output, _ = self.cross_attention(embeddings, projected_images, projected_images)

        # Decode attended output with GPT-2
        attended_output = attended_output.transpose(0, 1)  # Back to [batch_size, seq_len, hidden_size]
        outputs = self.gpt2(inputs_embeds=attended_output, attention_mask=attention_mask)
        return outputs.last_hidden_state


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
    batch_size = 2
    image_size = (3, 224, 224)
    seq_len = 10

    # Random images and captions
    dummy_images = torch.randn(batch_size, *image_size)  # Random images
    dummy_input_ids = torch.randint(0, 50257, (batch_size, seq_len))  # Random token IDs for GPT-2
    dummy_attention_mask = torch.ones_like(dummy_input_ids)  # Attention mask

    # Initialize the full model
    model = CaptionModel()

    # Preprocess images using the encoder's transform
    transform = model.encoder.transforms
    transformed_images = transform(dummy_images)

    # Forward pass
    output = model(transformed_images, dummy_input_ids, dummy_attention_mask)
    print("Output shape:", output.shape)  # Expected: [batch_size, seq_len, hidden_size]
