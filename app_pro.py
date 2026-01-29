# ==============================================================================
# MODULE: WEB INTERFACE (CLASSIFICATION MODE) + TTA (TEST-TIME AUGMENTATION)
# FRAMEWORK: STREAMLIT
# TECH: Averaging Predictions over Multiple Views
# ==============================================================================

import streamlit as st
import torch
import os
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

# ------------------------------------------------------------------------------
# 1. SYSTEM CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="AI Classification Pro (TTA Mode)")

# Dictionary quản lý đường dẫn các model
MODELS_CONFIG = {
    "BIRDS": {
        "name": "CUB-200 Bird Species",
        "path": "./model_birds_cub200", 
        "desc": "Fine-tuned ViT-Base on 200 bird classes (TTA Enabled)."
    },
    # "DOGS": {
    #     "name": "Stanford Dog Breeds",
    #     "path": "./model_dogs_stanford",
    #     "desc": "Fine-tuned ViT-Base on 120 dog breeds."
    # },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Các hàm xử lý)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_classification_model(model_path):
    """Tải Model và Processor."""
    if not os.path.exists(model_path):
        return None, None
    try:
        model = ViTForImageClassification.from_pretrained(model_path).to(DEVICE)
        processor = ViTImageProcessor.from_pretrained(model_path)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return None, None

def generate_augmented_images(original_image):
    """
    Tạo ra 5 phiên bản khác nhau của ảnh gốc để 'hội ý'.
    """
    # 1. Ảnh gốc
    images = [original_image]
    
    # 2. Lật ngang (Mirror) - Giúp nhận diện nếu chim quay đầu hướng khác
    images.append(ImageOps.mirror(original_image))
    
    # 3. Xoay nhẹ trái 15 độ
    images.append(original_image.rotate(15))
    
    # 4. Xoay nhẹ phải 15 độ
    images.append(original_image.rotate(-15))
    
    # 5. Crop trung tâm (Zoom vào 1 chút) - Giúp loại bỏ viền nhiễu
    width, height = original_image.size
    new_width = width * 0.85
    new_height = height * 0.85
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    images.append(original_image.crop((left, top, right, bottom)))
    
    return images

# ------------------------------------------------------------------------------
# 3. MAIN LOGIC
# ------------------------------------------------------------------------------
def main():
    st.title("🦅 AI Vision Pro: Multi-View Analysis")
    st.caption("Mode: Test-Time Augmentation (TTA) - Phân tích kỹ 5 góc độ")
    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Parameter Control")
        task_key = st.selectbox("Select Dataset:", list(MODELS_CONFIG.keys()))
        selected_config = MODELS_CONFIG[task_key]
        st.info(f"{selected_config['desc']}")
        st.success(f"Running on: {DEVICE.upper()}")

    # --- LOAD MODEL ---
    model, processor = load_classification_model(selected_config["path"])
    if model is None:
        st.error(f"CRITICAL ERROR: Model not found at `{selected_config['path']}`.")
        st.stop()

    # --- UI LAYOUT ---
    col_img, col_stat = st.columns([1, 1.5])

    with col_img:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image_source = Image.open(uploaded_file).convert("RGB")
            st.image(image_source, use_container_width=True, caption="Original Query")

    with col_stat:
        st.subheader("AI Consensus Result")
        
        if uploaded_file:
            with st.spinner("Đang phân tích 5 biến thể ảnh (TTA)..."):
                
                # BƯỚC 1: Tạo 5 ảnh biến thể
                aug_images = generate_augmented_images(image_source)
                
                # BƯỚC 2: Xử lý cả 5 ảnh cùng lúc (Batch Processing)
                inputs = processor(images=aug_images, return_tensors="pt").to(DEVICE)
                
                # BƯỚC 3: Dự đoán
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # BƯỚC 4: Tính trung bình cộng xác suất (Averaging)
                # outputs.logits có kích thước [5, 200] (5 ảnh, 200 lớp)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1) # Chuyển thành %
                
                # Tính trung bình dọc theo trục 0 (trục batch)
                avg_probs = torch.mean(probs, dim=0) 
                
                # Lấy Top 5 từ kết quả trung bình
                top_k_probs, top_k_indices = torch.topk(avg_probs, k=5)

            # BƯỚC 5: Hiển thị kết quả
            st.success("✅ Phân tích tổng hợp hoàn tất!")
            
            for i in range(5):
                confidence = top_k_probs[i].item()
                label_idx = top_k_indices[i].item()
                label_name = model.config.id2label[label_idx]
                
                # Logic hiển thị màu sắc dựa trên độ tin cậy
                if i == 0:
                    if confidence > 0.8: color = "green"
                    elif confidence > 0.5: color = "orange"
                    else: color = "red"
                    st.markdown(f":{color}[**TOP 1: {label_name}**]")
                else:
                    st.write(f"#{i+1}: {label_name}")
                
                st.progress(confidence)
                st.caption(f"Độ tin cậy trung bình: {confidence*100:.2f}%")

        else:
            st.info("Waiting for image...")

if __name__ == "__main__":
    main()