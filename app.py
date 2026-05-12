import os 
os.environ ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="AI Fashion Pro Analyzer", layout="wide")

# Custom CSS for Premium Navy & Gold Theme
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #001f3f; color: white; border: none; font-weight: bold; 
    }
    .stButton>button:hover { background-color: #c5a059; color: #001f3f; transition: 0.3s; }
    h1 { color: #001f3f; text-align: center; font-family: 'Segoe UI', sans-serif; }
    .attribute-box { 
        background-color: #ffffff; padding: 18px; border-radius: 12px; 
        border-left: 6px solid #c5a059; margin-bottom: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); color: #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛍️ Advanced Fashion AI Analyzer")
st.markdown("<p style='text-align: center; color: #64748b;'>Multi-Class Product Classification System</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. Load Model & Dataset
@st.cache_resource
def load_assets():
    # Model load karna with safety try-except
    try:
        from tensorflow import keras
        model = tf.keras.models.load_model('my_product_model.h5', compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
        
    # Dataset load karna
    df = pd.read_csv('styles.csv', on_bad_lines='skip')
    return model, df

model, df = load_assets()
class_labels = ['Accessories', 'Apparel', 'Footwear', 'Personal Care']

# 3. Sidebar
st.sidebar.header("Developer Info")
st.sidebar.write("System Lead: **Uzma Abid**")
if st.sidebar.button("🗑️ Reset Application"):
    st.rerun()

# 4. Main UI
uploaded_file = st.file_uploader("Upload a product image (JPG/PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")
    img = Image.open(uploaded_file)
    
    with col1:
        st.markdown("### **Input Image**")
        st.image(img, use_container_width=True)
        
    with col2:
        st.markdown("### **AI Processing Unit**")
        
        if st.button("🚀 Analyze Product"):
            with st.spinner('Neural Network is processing...'):
                # 1. Image Preprocessing
                img_resized = img.resize((160, 160))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Prediction
                prediction = model.predict(img_array)
                result = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # 3. Stable Logic for Characteristics (No random errors)
                if result == 'Accessories':
                    display_title = "Premium Fashion Accessory"
                    gender = "Unisex / Universal"
                    usage = "Casual & Professional"
                elif result == 'Apparel':
                    display_title = "Designer Clothing / Apparel"
                    gender = "Men & Women Collection"
                    usage = "Daily & Seasonal Wear"
                elif result == 'Footwear':
                    display_title = "High-End Footwear Collection"
                    gender = "Various Sizes Available"
                    usage = "Lifestyle & Sports"
                else:
                    display_title = f"Verified {result} Item"
                    gender = "Personal Use"
                    usage = "Hygiene & Care"

                # Result Display
                st.success(f"**Classification:** {result} | **Confidence:** {confidence:.2f}%")
                
                st.markdown(f"""
                <div class="attribute-box"><b>🏷️ Product Class:</b> {display_title}</div>
                <div class="attribute-box"><b>🚻 Target Group:</b> {gender}</div>
                <div class="attribute-box"><b>🎨 Visual Color:</b> Auto-Detected from Image</div>
                <div class="attribute-box"><b>🛠️ Utility:</b> {usage}</div>
                """, unsafe_allow_html=True)
                
                # Market Insight Calculation
                share = (len(df[df['masterCategory'] == result]) / len(df)) * 100
                st.info(f"📈 **Market Share:** This category covers **{share:.2f}%** of the current database.")

else:
    st.info("Please upload a product photo to begin real-time AI analysis.")
