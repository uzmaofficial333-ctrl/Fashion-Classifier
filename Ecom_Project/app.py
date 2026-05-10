import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd  # Nayi library data handle karne ke liye

# 1. Page Setup
st.set_page_config(page_title="E-commerce AI Analyzer", layout="wide")
st.title("🛍️ Smart Product Analyzer")

# 2. Model aur Data ko Load karna
@st.cache_resource
def load_assets():
    # Model load ho raha hai
    model = tf.keras.models.load_model('my_product_model.h5')
    # CSV file load ho rahi hai (on_bad_lines skip karein taake error na aaye)
    df = pd.read_csv('styles.csv', on_bad_lines='skip')
    return model, df

model, df = load_assets()
class_labels = ['Accessories', 'Apparel', 'Footwear', 'Personal Care']

st.success("Model aur Data successfully load ho gaye hain!")
# 3. Photo Upload aur Logic
uploaded_file = st.file_uploader("Product ki photo upload karein...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1]) # Screen ko 2 hisson mein baantna
    
    with col1:
        st.image(img, caption='Uploaded Product', use_container_width=True)
    
    # Image ko AI ke liye taiyar karna
    img_resized = img.resize((160, 160))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = class_labels[np.argmax(prediction)]

    with col2:
        st.header(f"Detected: {result}")
        
        # --- MARKET ANALYSIS ---
        st.subheader("📈 Market Analysis")
        total_items = len(df)
        cat_count = len(df[df['masterCategory'] == result])
        share = (cat_count / total_items) * 100
        st.write(f"Market Share of this Category is **{share:.2f}%** hai.")

        # --- PRODUCT DESCRIPTION ---
        st.subheader("📝 Smart Description")
        # CSV se is category ki misal nikalna
        details = df[df['masterCategory'] == result].iloc[0]
        st.write(f"**Gender:** {details['gender']}")
        st.write(f"**Season:** {details['season']}")
        st.write(f"**Usage:** {details['usage']}")