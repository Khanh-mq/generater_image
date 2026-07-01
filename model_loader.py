import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os

def load_model(model_path: str, lora_path: str = None):
    print("Loading SDXL FP16 VAE Fix to prevent color blob corruption...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

    print(f"Loading base model from {model_path}...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        vae=vae,
        torch_dtype=torch.float16,
    ).to("cuda")
    
    print("Applying EulerAncestralDiscreteScheduler (Required for NoobAI-XL)...")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    if not lora_path or not os.path.exists(lora_path):
        raise FileNotFoundError(f"❌ LỖI NGHIÊM TRỌNG: Không tìm thấy file LoRA tại {lora_path}. Bắt buộc phải có LoRA để chạy hệ thống!")
        
    print(f"Loading LoRA from {lora_path}...")
    weight_name = os.path.basename(lora_path)
    lora_dir = os.path.dirname(lora_path)
    pipe.load_lora_weights(lora_dir, weight_name=weight_name)
    
    print("Loading Qwen2-VL-2B for image validation...")
    qwen_id = "Qwen/Qwen2-VL-2B-Instruct"
    # Dùng torch.float16 để tiết kiệm VRAM
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        qwen_id, torch_dtype=torch.float16, device_map="cuda"
    )
    vlm_processor = AutoProcessor.from_pretrained(qwen_id)
    
    print("✅ All models loaded successfully!")
    return {
        "sdxl": pipe,
        "vlm_model": vlm_model,
        "vlm_processor": vlm_processor
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