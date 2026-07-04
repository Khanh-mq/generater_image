import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
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
    
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from {lora_path}...")
        weight_name = os.path.basename(lora_path)
        lora_dir = os.path.dirname(lora_path)
        pipe.load_lora_weights(lora_dir, weight_name=weight_name)
    else:
        print("Bỏ qua bước load LoRA vì không tìm thấy file hoặc cấu hình không yêu cầu.")
    
    print("✅ All models loaded successfully!")
    return {
        "sdxl": pipe
    }

def generate_image(pipe, prompt, negative_prompt, width, height, num_steps, guidance_scale, seed, lora_weight):
    generator = torch.Generator("cuda").manual_seed(seed)
    
    kwargs = {}
    # Chỉ truyền scale cho LoRA nếu lora_weight > 0
    if lora_weight is not None and lora_weight > 0:
        kwargs["cross_attention_kwargs"] = {"scale": lora_weight}

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **kwargs
    ).images[0]
    
    return image