import os
import base64
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model, generate_image

app = FastAPI(title="LoRA Image Generation API")

# Cấu hình đường dẫn model
# Trên RunPod, bạn có thể mount volume vào thư mục này để chứa model/lora
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/noobai_xl_v1.1.safetensors")
LORA_PATH = os.getenv("LORA_PATH", "/app/models/lora_lung_mat.safetensors")

# Load model global khi khởi động
pipe = None

@app.on_event("startup")
async def startup_event():
    global pipe
    try:
        # Tải model vào RAM/VRAM ngay khi API khởi động
        pipe = load_model(MODEL_PATH, LORA_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Note: In production you might want to stop startup if model fails

class PromptItem(BaseModel):
    shot_id: str
    prompt_text: str

class GenerationRequest(BaseModel):
    prompts: List[PromptItem]
    negative_prompt: Optional[str] = "realistic, 3d, photorealistic, blurry, multiple characters, character sheet, model sheet, reference sheet, collage, grid"
    width: Optional[int] = 1280
    height: Optional[int] = 720
    num_steps: Optional[int] = 25
    guidance_scale: Optional[float] = 11.0
    seed: Optional[int] = 42
    lora_weight: Optional[float] = 0.8

class ImageResult(BaseModel):
    shot_id: str
    image_base64: str
    seed: int
    prompt: str

class GenerationResponse(BaseModel):
    results: List[ImageResult]

@app.post("/generate", response_model=GenerationResponse)
async def api_generate(request: GenerationRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to load.")
    
    results = []
    # Xử lý tuần tự từng prompt trong batch
    for i, item in enumerate(request.prompts):
        # Có thể dùng seed khác nhau cho mỗi prompt trong batch nếu muốn
        current_seed = request.seed + i
        
        try:
            image = generate_image(
                pipe=pipe,
                prompt=item.prompt_text,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                seed=current_seed,
                lora_weight=request.lora_weight
            )
            
            # Chuyển ảnh PIL sang Base64 string để trả về
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            results.append(ImageResult(
                shot_id=item.shot_id,
                image_base64=img_str,
                seed=current_seed,
                prompt=item.prompt_text
            ))
            
        except Exception as e:
            print(f"Error generating image for shot_id '{item.shot_id}': {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    return GenerationResponse(results=results)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipe is not None}