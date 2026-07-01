import torch
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_model(model_path: str, lora_path: str = None):
    print(f"Loading base model from {model_path}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    if not lora_path or not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file LoRA tại {lora_path}. Bắt buộc phải có LoRA để chạy hệ thống!")
        
    print(f"Loading LoRA from {lora_path}...")
    weight_name = os.path.basename(lora_path)
    lora_dir = os.path.dirname(lora_path)
    pipe.load_lora_weights(lora_dir, weight_name=weight_name)
    
    print("Loading Moondream2 for image validation...")
    moondream_id = "vikhyatk/moondream2"
    moondream_revision = "2024-08-26"
    md_model = AutoModelForCausalLM.from_pretrained(
        moondream_id, trust_remote_code=True, revision=moondream_revision, torch_dtype=torch.float16
    ).to("cuda")
    md_tokenizer = AutoTokenizer.from_pretrained(moondream_id, revision=moondream_revision)
    
    print("✅ All models loaded successfully!")
    return {
        "sdxl": pipe,
        "md_model": md_model,
        "md_tokenizer": md_tokenizer
    }

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