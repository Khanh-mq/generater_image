import requests
import base64
from io import BytesIO
from PIL import Image

# Thay bằng IP hoặc Domain mà RunPod cung cấp
API_URL = "https://7951qnpfhlx13o-8000.proxy.runpod.net/generate"

# Bộ khung Prompt Nhất quán (Không bao giờ thay đổi)
BASE_PROMPT = "(lung_mat_char:1.1), 1boy, anthro, honey badger, (white stripe on head:1.2), grey body fur, black face mask, yellow eyes, masterpiece, best quality, flat color, kurzgesagt style"

# Trang phục mặc định của nhân vật
OUTFIT = "wearing casual red jacket, blue jeans"

# Dữ liệu bạn muốn gửi lên server
payload = {
    "prompts": [
        {
            "shot_id": "shot_01", 
            "prompt_text": f"{BASE_PROMPT}, {OUTFIT}, walking home, carrying backpack, tired expression, sunset background, residential street"
        },
        {
            "shot_id": "shot_02", 
            "prompt_text": f"{BASE_PROMPT}, {OUTFIT}, drinking coffee, holding mug, sitting at table, morning sunlight, cozy room"
        }
    ],
    "negative_prompt": "realistic, 3d, photorealistic, blurry, multiple characters, extra limbs, bad anatomy, bad hands, missing fingers, human face, realistic animal, deformed, ugly, messy lines, text, signature, watermark",
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
