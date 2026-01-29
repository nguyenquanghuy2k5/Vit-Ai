# ==============================================================================
# MODULE: AI INFERENCE SYSTEM (MULTI-DOMAIN)
# DOMAINS: BIRDS (CUB-200), DOGS (STANFORD), FOOD (VIETNAMESE)
# ==============================================================================

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import os

# ------------------------------------------------------------------------------
# 1. SYSTEM CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Multi-Domain Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CẤU HÌNH ĐƯỜNG DẪN MODEL (QUAN TRỌNG: Lưu đúng các folder này tại ổ D)
MODELS_CONFIG = {
    "BIRDS": {
        "id": "BIRDS",
        "name": "CUB-200-2011 (Bird Species)",
        "path": "D:/AI_Projects/model_birds_cub200_final",
        "desc": "Standard Benchmark Split (200 Classes)"
    },
    "DOGS": {
        "id": "DOGS",
        "name": "Stanford Dogs (Dog Breeds)",
        "path": "D:/AI_Projects/model_dogs_stanford_benchmark",  # <--- DỰ KIẾN
        "desc": "Standard Benchmark (120 Classes)"
    },
    "FOOD": {
        "id": "FOOD",
        "name": "Vietnamese Food (Cuisine)",
        "path": "D:/AI_Projects/model_food_vietnam",             # <--- DỰ KIẾN
        "desc": "Vietnamese Traditional Dishes (30 Classes)"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# 2. CORE FUNCTIONS
# ------------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    """
    Load model resources with robust error handling.
    """
    # 1. Check if directory exists
    if not os.path.exists(model_path):
        return None, None, f"Directory not found: {model_path}"

    try:
        # 2. Load Model
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        
        # 3. Load Processor
        try:
            processor = ViTImageProcessor.from_pretrained(model_path)
        except OSError:
            # Fallback to Google Hub
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            
        return model, processor, "SUCCESS"
        
    except Exception as e:
        return None, None, str(e)

def predict(image, model, processor):
    """
    Execute inference pipeline.
    """
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # Get Top 5 Predictions
        top_probs, top_indices = torch.topk(probs, k=5)
        
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

# ------------------------------------------------------------------------------
# 3. USER INTERFACE
# ------------------------------------------------------------------------------
def main():
    st.title("AI Visual Recognition System")
    st.markdown("### Fine-Grained Visual Categorization")
    st.divider()

    # --- SIDEBAR: MODEL SELECTION ---
    with st.sidebar:
        st.header("System Configuration")
        
        # Dropdown to switch between BIRDS / DOGS / FOOD
        domain_choice = st.selectbox("Select Domain", list(MODELS_CONFIG.keys()))
        selected_config = MODELS_CONFIG[domain_choice]
        
        st.info(f"**Target:** {selected_config['name']}\n\n**Description:** {selected_config['desc']}")
        st.text(f"Hardware Acceleration: {DEVICE.upper()}")

    # --- MAIN AREA ---
    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN: INPUT
    with col1:
        st.subheader("Input Data")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Query Image", use_container_width=True)

    # RIGHT COLUMN: OUTPUT
    with col2:
        st.subheader("Inference Analysis")
        
        if uploaded_file:
            # Load selected model
            with st.spinner(f"Loading {domain_choice} Model..."):
                model, processor, status = load_model(selected_config["path"])

            # Check status
            if status != "SUCCESS":
                st.warning(f"Model not ready for {domain_choice}.")
                st.error(f"Error Detail: {status}")
                st.info("Please train this model and save it to the path specified in Config.")
            else:
                # Run Prediction
                top_probs, top_indices = predict(image, model, processor)
                
                # Render Results
                st.markdown("#### Top 5 Confidence Scores")
                
                for i in range(5):
                    prob = top_probs[i]
                    label_idx = top_indices[i]
                    # Handle label names
                    if hasattr(model.config, 'id2label'):
                        label_name = model.config.id2label[label_idx]
                    else:
                        label_name = f"Class ID {label_idx}"
                    
                    # Visualization
                    c1, c2, c3 = st.columns([3, 5, 2])
                    with c1:
                        if i == 0:
                            st.markdown(f"**{label_name}**")
                        else:
                            st.write(f"{label_name}")
                    with c2:
                        st.progress(float(prob))
                    with c3:
                        st.write(f"{prob*100:.2f}%")

if __name__ == "__main__":
    main()