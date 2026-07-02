import os
import base64
from io import BytesIO
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model, generate_image
import random

app = FastAPI(title="LoRA Image Generation API")

# Cấu hình đường dẫn model
# Trên RunPod, bạn có thể mount volume vào thư mục này để chứa model/lora
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/Juggernaut-XL_v9.safetensors")
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
    validation_query: Optional[str] = None

class GenerationRequest(BaseModel):
    prompts: List[PromptItem]
    negative_prompt: Optional[str] = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, blurry"
    width: Optional[int] = 1280
    height: Optional[int] = 720
    num_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 6.0
    seed: Optional[int] = 42
    lora_weight: Optional[float] = 0.0

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
        current_seed = request.seed + i
        current_prompt = item.prompt_text
        current_cfg = request.guidance_scale
        
        max_retries = 3
        best_image = None
        best_seed = current_seed
        
        for attempt in range(max_retries):
            try:
                print(f"[{item.shot_id}] Attempt {attempt+1}/{max_retries} with seed {current_seed}")
                image = generate_image(
                    pipe=pipe["sdxl"],
                    prompt=current_prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_steps=request.num_steps,
                    guidance_scale=current_cfg,
                    seed=current_seed,
                    lora_weight=request.lora_weight
                )
                
                best_image = image
                best_seed = current_seed
                
                if item.validation_query:
                    # Chạy Qwen2-VL để kiểm tra
                    vlm_model = pipe["vlm_model"]
                    vlm_processor = pipe["vlm_processor"]
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": f"{item.validation_query} Answer strictly with 'Yes' or 'No'."},
                            ],
                        }
                    ]
                    
                    from qwen_vl_utils import process_vision_info
                    text_prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = vlm_processor(
                        text=[text_prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to("cuda")
                    
                    generated_ids = vlm_model.generate(**inputs, max_new_tokens=5)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    answer = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    
                    print(f"Validation: '{item.validation_query}' -> Answer: {answer}")
                    
                    if "yes" in answer.lower():
                        print(f"✅ Image passed validation!")
                        break # Ảnh tốt, thoát vòng lặp retry
                    else:
                        print(f"❌ Image failed validation. Retrying...")
                        current_seed = random.randint(1, 9999999)
                        if attempt == 0:
                            current_cfg = max(5.0, current_cfg - 2.0)
                        elif attempt == 1:
                            current_prompt = f"({current_prompt}:1.2)"
                else:
                    break # Không cần validate, thoát vòng lặp luôn
                
            except Exception as e:
                print(f"Error generating image for shot_id '{item.shot_id}': {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=str(e))
                current_seed = random.randint(1, 9999999)

        if best_image:
            # Chuyển ảnh PIL sang Base64 string để trả về
            buffered = BytesIO()
            best_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            results.append(ImageResult(
                shot_id=item.shot_id,
                image_base64=img_str,
                seed=best_seed,
                prompt=current_prompt
            ))
            
    return GenerationResponse(results=results)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipe is not None and "sdxl" in pipe}