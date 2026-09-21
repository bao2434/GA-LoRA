import torch
from transformers import CLIPProcessor
CLIP_MODEL = {
    "B32": "openai/clip-vit-base-patch32",
    "B16": "openai/clip-vit-base-patch16",
    "L14": "openai/clip-vit-large-patch14",
}
class CLIP_Processor:
    def __init__(self, model="B16", cache_dir="./models"):
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL[model], cache_dir=cache_dir)

    def __call__(self, *args, **kwargs):       
        output = self.processor(*args, **kwargs)
        return output
    
    def process_image(self, x):
        if isinstance(x, dict):
            image = x["image"][[3, 2, 1], :, :] 
            inputs = self.processor(images=image, return_tensors="pt")
            x["image"] = inputs["pixel_values"].squeeze(0)
            return x
        else:
            inputs = self.processor(images=x, return_tensors="pt")
            pixel_values = inputs['pixel_values'] 
            pixel_values = pixel_values.squeeze(0)
            return pixel_values