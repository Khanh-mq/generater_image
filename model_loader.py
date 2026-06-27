import torch
from diffusers import StableDiffusionXLPipeline
import os

def load_model(model_path: str, lora_path: str = None):
    print(f"Loading base model from {model_path}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from {lora_path}...")
        # Lấy tên file để làm weight_name
        weight_name = os.path.basename(lora_path)
        lora_dir = os.path.dirname(lora_path)
        pipe.load_lora_weights(lora_dir, weight_name=weight_name)
    
    print("Model and LoRA loaded successfully!")
    return pipe

def generate_image(pipe, prompt, negative_prompt, width, height, num_steps, guidance_scale, seed, lora_weight):
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        cross_attention_kwargs={"scale": lora_weight}
    ).images[0]
    
    return image