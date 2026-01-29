# ==============================================================================
# MODULE: SINGLE IMAGE PREDICTION (INFERENCE)
# DESCRIPTION: Dự đoán nhãn cho 1 ảnh bất kỳ (Từ URL hoặc File máy tính)
# ==============================================================================

import torch
import requests
from PIL import Image
from io import BytesIO
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification
import os

# ==============================================================================
# 1. CẤU HÌNH NGƯỜI DÙNG (USER CONFIGURATION)
# ==============================================================================
# Chọn Mode tương ứng với Model bạn muốn test
# Options: "BIRDS" | "DOGS" | "FOOD"
TASK_MODE = "BIRDS" 

# --- NHẬP LINK ẢNH HOẶC ĐƯỜNG DẪN FILE CẦN TEST TẠI ĐÂY ---
# Cách 1: Dùng Link Online (URL)
IMAGE_SOURCE = "D:/anh1.jpg"

# Cách 2: Dùng ảnh trong máy (Bỏ comment dòng dưới và điền đường dẫn)
# IMAGE_SOURCE = "C:/Users/Admin/Desktop/test_bird.jpg"

# ==============================================================================
# 2. THIẾT LẬP HỆ THỐNG
# ==============================================================================
if TASK_MODE == "BIRDS":
    MODEL_PATH = "./model_birds_cub200"
elif TASK_MODE == "DOGS":
    MODEL_PATH = "./model_dogs_stanford"
elif TASK_MODE == "FOOD":
    MODEL_PATH = "./model_vietnamese_food"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(source):
    """Hàm hỗ trợ đọc ảnh từ cả URL và Local File"""
    try:
        if source.startswith("http"): # Nếu là URL
            print(f"--> [IO] Đang tải ảnh từ Internet...")
            response = requests.get(source, timeout=10)
            image = Image.open(BytesIO(response.content))
        else: # Nếu là File trong máy
            print(f"--> [IO] Đang đọc file từ ổ cứng...")
            if not os.path.exists(source):
                raise FileNotFoundError(f"Không tìm thấy file: {source}")
            image = Image.open(source)
        
        # Chuyển sang chuẩn RGB (đề phòng ảnh PNG 4 kênh hoặc ảnh xám)
        return image.convert("RGB")
    except Exception as e:
        print(f"[LỖI] Không thể đọc ảnh: {e}")
        exit()

def main():
    print("="*60)
    print(f"PREDICTION SESSION: {TASK_MODE} | DEVICE: {DEVICE.upper()}")
    print("="*60)

    # 1. Load Model & Processor
    print(f"[1/3] Loading Model from: {MODEL_PATH}...")
    try:
        model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        model.eval() # Quan trọng: Chuyển sang chế độ dự đoán
    except Exception as e:
        print(f"[CRITICAL ERROR] Không load được model. Bạn đã train xong chưa?\nChi tiết: {e}")
        exit()

    # 2. Xử lý ảnh (Pre-processing)
    print(f"[2/3] Processing Image...")
    image = load_image(IMAGE_SOURCE)
    
    # Biến đổi ảnh thành Tensor để đưa vào mô hình
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # 3. Dự đoán (Inference)
    print(f"[3/3] Analyzing...")
    with torch.no_grad(): # Tắt tính toán gradient để nhẹ máy
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1) # Chuyển logits -> Xác suất %
        
        # Lấy Top 3 kết quả cao nhất
        top_probs, top_indices = torch.topk(probs, k=3)
        
    # 4. Hiển thị kết quả
    print("\n" + "="*40)
    print("KẾT QUẢ PHÂN TÍCH (TOP 3)")
    print("="*40)
    
    for i in range(3):
        score = top_probs[0][i].item() * 100
        class_idx = top_indices[0][i].item()
        class_name = model.config.id2label[class_idx]
        
        # In ra đẹp mắt
        print(f"HẠNG {i+1}: {class_name:<30} | {score:.2f}%")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    main()