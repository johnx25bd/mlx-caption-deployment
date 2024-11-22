import logging
import torch
import PIL
from PIL import Image

from services.load_weights import load_weights
from models.vision_transformer import CaptionModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CaptionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = CaptionModel(eval=True, logger=logger, inference=True).to(self.device)
        state_dict = load_weights(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def prepare_inference_input(self, pil_image):
        logger.debug("Preparing inference input")
        try:
            transformed_image = self.model.encoder.transforms(pil_image)
            transformed_image = transformed_image.unsqueeze(0).to(self.device)

            tokenizer = self.model.decoder.tokenizer
            start_prompt = tokenizer.bos_token
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
        except Exception as e:
            logger.error(f"Error in prepare_inference_input: {e}")
            raise

    def predict_caption(self, pil_image, max_length=15):
        try:
            # Resize image if needed
            w, h = pil_image.size
            
            if max(w, h) > 224:
                ratio = 224.0 / max(w, h)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                logger.debug(f"Resizing image to: {new_size}")
                pil_image = pil_image.resize(new_size, PIL.Image.Resampling.LANCZOS)

            image_input, caption_input, mask = self.prepare_inference_input(pil_image)
            predicted_caption_tokens = []
            
            with torch.no_grad():
                try:
                    for i in range(max_length):
                        logger.debug(f"Generation step {i}")
                        logits = self.model(image_input, caption_input, mask)
                        next_token_logits = logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, -1)

                        if next_token.item() == self.model.decoder.tokenizer.eos_token_id:
                            logger.debug("Found EOS token, stopping generation")
                            break

                        predicted_caption_tokens.append(next_token.item())
                        caption_input = torch.cat([caption_input, next_token.unsqueeze(0)], dim=1)
                        mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.bool, device=self.device)], dim=1)

                except RuntimeError as e:
                    logger.error(f"Runtime error during generation: {e}")
                    torch.cuda.empty_cache()
                    raise

            if predicted_caption_tokens:
                caption = self.model.decoder.tokenizer.decode(predicted_caption_tokens, skip_special_tokens=False)
                return caption
            else:
                logger.warning("No tokens generated")
                return "No caption generated"

        except Exception as e:
            logger.error(f"Error in predict_caption: {e}", exc_info=True)
            return f"Error generating caption: {str(e)}"