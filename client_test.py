import requests
import base64
from io import BytesIO
from PIL import Image

# Thay bằng IP hoặc Domain mà RunPod cung cấp
API_URL = "http://YOUR_RUNPOD_IP:8000/generate"

# Dữ liệu bạn muốn gửi lên server
payload = {
    "prompts": [
        {"shot_id": "shot_01", "prompt_text": "lung_mat_char, honey badger, anthro, walking home, carrying backpack, sunset background, residential street, tired expression"},
        {"shot_id": "shot_02", "prompt_text": "lung_mat_char, honey badger, anthro, drinking coffee, morning sunlight, cozy room"}
    ],
    "negative_prompt": "realistic, 3d, photorealistic, blurry, multiple characters",
    "width": 1280,
    "height": 720,
    "num_steps": 25,
    "guidance_scale": 11.0,
    "seed": 42,
    "lora_weight": 0.8
}

print("Gửi request lên server... Vui lòng đợi (có thể mất vài chục giây)")
response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    data = response.json()
    results = data.get("results", [])
    
    for i, item in enumerate(results):
        b64_str = item["image_base64"]
        shot_id = item["shot_id"]
        seed = item["seed"]
        prompt = item["prompt"]
        
        # Decode base64 thành bytes
        img_data = base64.b64decode(b64_str)
        # Chuyển bytes thành ảnh PIL
        image = Image.open(BytesIO(img_data))
        
        # Lưu ảnh về laptop kèm theo số seed để dễ quản lý
        save_path = f"result_{shot_id}_seed_{seed}.png"
        image.save(save_path)
        print(f"✅ Đã tải và lưu ảnh: {save_path} (Seed: {seed})")
else:
    print(f"❌ Lỗi {response.status_code}: {response.text}")
