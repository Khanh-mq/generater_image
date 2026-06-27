# Xóa torchvision cũ bị conflict
import subprocess, sys

print("🗑️  Xóa torchvision cũ...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True)

import torch
cuda_ver = torch.version.cuda.replace(".", "")[:3]  # vd: "130"
torch_ver = torch.__version__
print(f"torch   : {torch_ver}")
print(f"CUDA    : {torch.version.cuda}")
print(f"cuda tag: cu{cuda_ver}")

print(f"\n⬇️  Cài torchvision nightly cho cu{cuda_ver}...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "torchvision",
     "--index-url", f"https://download.pytorch.org/whl/nightly/cu{cuda_ver}"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "")
print(result.stderr[-300:] if result.stderr else "")

print("\n✅ Xong! Bây giờ: Runtime → Restart session → chạy tiếp Bước 2")

# --- 

!pip install -q numpy==2.0.0
!pip install -q diffusers==0.31.0 transformers==4.44.0 peft==0.14.0 accelerate==0.33.0

# --- 

# Kiểm tra import được không
import torch
from diffusers import StableDiffusionXLPipeline
print(f"✅ torch      : {torch.__version__}")
import diffusers, transformers
print(f"✅ diffusers  : {diffusers.__version__}")
print(f"✅ transformers: {transformers.__version__}")
print(f"✅ CUDA       : {torch.cuda.is_available()}")
print(f"✅ GPU        : {torch.cuda.get_device_name(0)}")

# --- 

from google.colab import drive
drive.mount('/content/drive')

import os, glob

LORA_DIR = "/content/drive/MyDrive/Loras/lung_mat/output/"
lora_files = sorted(glob.glob(f"{LORA_DIR}*.safetensors"))

if lora_files:
    print(f"✅ Tìm thấy {len(lora_files)} file LoRA:")
    for f in lora_files:
        size = os.path.getsize(f) / 1024**2
        print(f"  📄 {os.path.basename(f)} ({size:.0f} MB)")
    LATEST_LORA = os.path.basename(lora_files[3])
    print(f"\n🎯 Sẽ dùng: {LATEST_LORA}")
else:
    print("❌ Không tìm thấy file LoRA trong Drive!")
    print(f"   Kiểm tra lại đường dẫn: {LORA_DIR}")

# --- 

!apt install -y aria2 -q

import os
os.makedirs("/content/drive/MyDrive/models", exist_ok=True)
!aria2c "https://huggingface.co/Laxhar/noobai-XL-1.1/resolve/main/NoobAI-XL-v1.1.safetensors" \
    --console-log-level=warn \
    -c -s 16 -x 16 -k 10M \
    -d "/content/drive/MyDrive/models" \
    -o "noobai_xl_v1.1.safetensors"

print("✅ Download xong!")

print("✅ Download xong!")

# --- 

from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    "/content/drive/MyDrive/models/noobai_xl_v1.1.safetensors",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights(LORA_DIR, weight_name=LATEST_LORA)
print("✅ Sẵn sàng generate!")

# --- 

# import torch
# from diffusers import StableDiffusionXLPipeline

# print("⬇️  Loading Illustrious XL...")
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "/content/drive/MyDrive/models/noobai_xl.safetensors",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
# ).to("cuda")
# print("✅ Model loaded!")

# print(f"\n🔗 Loading LoRA: {LATEST_LORA}...")
# pipe.load_lora_weights(LORA_DIR, weight_name=LATEST_LORA)
# print("✅ LoRA loaded!")

# --- 

# ============================================
# ⚙️ CHỈNH THÔNG SỐ Ở ĐÂY
# ============================================

PROMPT = (
    "lung_mat_char, honey badger, anthro, walking home, carrying backpack, sunset background, residential street, tired expression, flat color, thick black outline, cartoon, simple illustration, 2d, solo, full body, small eyes, short snout"
)

NEGATIVE = (
    "realistic, 3d, photorealistic, blurry, multiple characters, character sheet, model sheet, reference sheet, collage, grid"
)

LORA_WEIGHT    = 0.8   # độ mạnh LoRA: 0.6 - 1.0
NUM_STEPS      = 25    # bước inference: 20-30
GUIDANCE_SCALE = 11    # CFG scale: 5-9
SEED           = 42    # đổi số này để ra ảnh khác
WIDTH          = 1280
HEIGHT         = 720

# ============================================

import torch
from IPython.display import display

generator = torch.Generator("cuda").manual_seed(SEED)

print("🎨 Đang generate ảnh...")
image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
    cross_attention_kwargs={"scale": LORA_WEIGHT}
).images[0]

display(image)

# Lưu vào Drive
save_path = f"/content/drive/MyDrive/lung_mat_seed{SEED}.png"
image.save(save_path)
print(f"\n✅ Đã lưu: {save_path}")

# --- 

import torch
from IPython.display import display
import os

NUM_IMAGES = 4  # số ảnh muốn tạo

print(f"🎨 Generating {NUM_IMAGES} ảnh...\n")

for i in range(NUM_IMAGES):
    seed = SEED + i
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        cross_attention_kwargs={"scale": LORA_WEIGHT}
    ).images[0]

    display(image)
    save_path = f"/content/drive/MyDrive/lung_mat_seed{seed}.png"
    image.save(save_path)
    print(f"✅ Ảnh {i+1}/{NUM_IMAGES} — seed={seed} — lưu tại Drive")

# --- 

