import streamlit as st
import torch
import time
import pandas as pd
import plotly.express as px
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from streamlit_cropper import st_cropper 

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Phân loại Ảnh (Demo)",
    layout="wide"
)

# --- SỬA LỖI CSS: Xóa các đoạn ép màu nền gây lỗi ---
st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    
    /* Chỉ chỉnh nút bấm cho đẹp, không can thiệp màu chữ hệ thống */
    .stButton>button {
        width: 100%; 
        border-radius: 4px; 
        height: 3em; 
        font-weight: bold;
        background-color: #0068c9; /* Màu xanh chuẩn Streamlit */
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0053a0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# KHO CHỨA MODEL (GIỮ NGUYÊN ĐƯỜNG DẪN CỦA BẠN)
MODEL_REGISTRY = {
    "DOGS": {
        "title": "Dữ liệu: Stanford Dogs (120 Lớp)",
        "models": {
            "ViT": {
                "path": "D:/AI_Projects/model_dogs_stanford_benchmark",
                "acc": "89.7%",
                "desc": "Kiến trúc: Vision Transformer (ViT-Base)"
            },
            "ResNet": {
                "path": "D:/AI_Projects/model_dogs_resnet50_baseline",
                "acc": "78.2%",
                "desc": "Kiến trúc: ResNet-50 (CNN Baseline)"
            }
        }
    },
    "BIRDS": {
        "title": "Dữ liệu: CUB-200-2011 (200 Lớp)",
        "models": {
            "ViT": {
                "path": "D:/AI_Projects/model_birds_cub200_final",
                "acc": "84.3%",
                "desc": "Kiến trúc: Vision Transformer (ViT-Base)"
            },
            "ResNet": {
                "path": "D:/AI_Projects/model_birds_resnet50_baseline",
                "acc": "51.0%",
                "desc": "Kiến trúc: ResNet-50 (CNN Baseline)"
            }
        }
    }
}

# ==============================================================================
# 2. HÀM LOAD MODEL
# ==============================================================================
@st.cache_resource
def load_model_pipeline(path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForImageClassification.from_pretrained(path).to(device)
        return processor, model, device, None
    except Exception as e:
        return None, None, None, str(e)

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================
st.sidebar.header("Cấu hình Tham số")
st.sidebar.markdown("---")

domain = st.sidebar.selectbox(
    "Chọn Tập dữ liệu:",
    ["DOGS", "BIRDS"],
    format_func=lambda x: "Stanford Dogs" if x == "DOGS" else "CUB-200 Birds"
)

model_type = st.sidebar.radio(
    "Chọn Mô hình:",
    ["ViT", "ResNet"],
    help="ViT: Mô hình đề xuất | ResNet: Mô hình đối chứng"
)

current_info = MODEL_REGISTRY[domain]["models"][model_type]
st.sidebar.markdown("---")
st.sidebar.info(f"Độ chính xác (Acc): {current_info['acc']}")
st.sidebar.caption(current_info['desc'])

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================
st.title("Demo Phân loại Ảnh Đa miền")
st.subheader(MODEL_REGISTRY[domain]["title"])
st.markdown("---")

col1, col2 = st.columns([1, 1.2], gap="large")

# --- CỘT TRÁI ---
with col1:
    st.markdown("##### 1. Tiền xử lý (Cắt ảnh)")
    st.caption("Kéo khung đỏ để chọn vùng đối tượng.")
    
    uploaded_file = st.file_uploader("Tải ảnh lên (JPG, PNG)", type=['jpg', 'png', 'jpeg'])
    
    cropped_image = None
    if uploaded_file:
        img_file = Image.open(uploaded_file).convert('RGB')
        
        cropped_image = st_cropper(
            img_file,
            realtime_update=True,
            box_color='#FF0000',
            aspect_ratio=None,
            should_resize_image=True
        )
        st.text("Preview: Ảnh đầu vào cho Model")
    else:
        st.info("Vui lòng tải ảnh lên để bắt đầu.")

# --- CỘT PHẢI ---
with col2:
    st.markdown("##### 2. Kết quả Dự đoán")
    
    if cropped_image:
        processor, model, device, error = load_model_pipeline(current_info["path"])
        
        if error:
            st.error(f"Lỗi Model: {error}")
            st.stop()

        st.image(cropped_image, width=150, caption="Input (Cropped)")
        
        if st.button("Thực hiện Phân loại"):
            with st.spinner("Đang xử lý..."):
                start_time = time.time()
                
                inputs = processor(images=cropped_image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k)
                
                end_time = time.time()
                inference_time = end_time - start_time
                
                top_labels = [model.config.id2label[idx.item()] for idx in top_indices]
                top_scores = [val.item() * 100 for val in top_probs]
                
                best_label = top_labels[0]
                best_score = top_scores[0]

                # --- HIỂN THỊ METRICS (KHÔNG CÓ BACKGROUND CỨNG) ---
                # Streamlit sẽ tự động chọn màu chữ đen/trắng tùy theo theme của máy bạn
                m1, m2 = st.columns(2)
                m1.metric("Nhãn dự đoán (Top-1)", best_label)
                m1.metric("Độ tin cậy", f"{best_score:.2f}%")
                
                m2.metric("Thời gian suy luận", f"{inference_time:.4f}s")
                m2.metric("Thiết bị xử lý", device.upper())
                
                st.markdown("---")
                
                # --- BIỂU ĐỒ ---
                st.markdown("##### Phân phối xác suất (Top-5)")
                
                chart_data = pd.DataFrame({
                    "Lớp": top_labels[::-1], 
                    "Xác suất": top_scores[::-1]
                })
                
                fig = px.bar(
                    chart_data, 
                    x="Xác suất", 
                    y="Lớp", 
                    orientation='h',
                    text_auto='.2f',
                )
                
                fig.update_layout(
                    height=300, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Độ tin cậy (%)",
                    yaxis_title=None,
                    font=dict(size=13) # Tăng cỡ chữ biểu đồ cho dễ đọc
                )
                fig.update_traces(marker_color='#0068c9') # Màu xanh tương đồng với nút bấm
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Đang chờ dữ liệu đầu vào...")