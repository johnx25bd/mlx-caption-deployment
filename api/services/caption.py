import PIL
import torch
from models.vision_transformer import CaptionModel

class CaptionService:
    def __init__(self):
        super().__init__()
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.model = CaptionModel().to(self.device)
        self.model.eval()

    def prepare_inference_input(self, pil_image):
        transformed_image = self.model.encoder.transforms(pil_image).unsqueeze(0).to(self.device)

        tokenizer = self.model.decoder.tokenizer
        start_prompt = tokenizer.bos_token
        print("start pr", start_prompt)

        tokens = tokenizer(
            start_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1
        )

        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.bool().to(self.device)
        return transformed_image, input_ids, attention_mask

    def predict_caption(self, pil_image, max_length=50):
        image_input, caption_input, attention_mask = self.prepare_inference_input(pil_image)
        predicted_caption_tokens = []
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.model(image_input, caption_input, attention_mask) # batch_size, seq_length, vocab_size
                # each time, grab the logits final token in the generated sequence
                next_token_logits = logits[:, -1, :] # batch_size, 1, vocab_size
                next_token = torch.argmax(next_token_logits, -1) # vocab_size

                # why does this work? it's getting an item 
                if next_token.item() == self.model.decoder.tokenizer.eos_token_id:
                    break
                caption_input = torch.cat([caption_input, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.bool, device=self.device)], dim=1)
                
                predicted_caption_tokens.append(next_token.item())
        caption = self.model.decoder.tokenizer.decode(predicted_caption_tokens, skip_special_tokens=True)
        return caption

if __name__ == "__main__":
    caption_service = CaptionService()
    image = PIL.Image.open("./dog.png").convert("RGB")
    caption = caption_service.predict_caption(image)
    print(f"caption:{caption}") 



    
