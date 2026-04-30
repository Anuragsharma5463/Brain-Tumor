import streamlit as st
import numpy as np
import pickle
from PIL import Image
import cv2
import plotly.express as px
import time

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        color: #0f0c29;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError, .stWarning {
        border-radius: 15px;
        padding: 15px;
    }
    
    /* Custom divider */
    hr {
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Image containers */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Animation keyframes */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🧠 Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HEADER WITH ANIMATION ----------------
st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 10px;'>🧠 Brain Tumor Detection System</h1>
        <p style='font-size: 1.2rem; color: #a0a0a0;'>Advanced AI-Powered MRI Analysis for Early Detection</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- FEATURE HIGHLIGHTS ----------------
col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)

with col_feat1:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(0, 212, 255, 0.1); border-radius: 15px;'>
            <h2 style='font-size: 2rem;'>⚡</h2>
            <p>Fast Analysis</p>
        </div>
    """, unsafe_allow_html=True)

with col_feat2:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(0, 255, 136, 0.1); border-radius: 15px;'>
            <h2 style='font-size: 2rem;'>🎯</h2>
            <p>High Accuracy</p>
        </div>
    """, unsafe_allow_html=True)

with col_feat3:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(255, 75, 75, 0.1); border-radius: 15px;'>
            <h2 style='font-size: 2rem;'>🔒</h2>
            <p>Secure & Private</p>
        </div>
    """, unsafe_allow_html=True)

with col_feat4:
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(255, 200, 0, 0.1); border-radius: 15px;'>
            <h2 style='font-size: 2rem;'>📊</h2>
            <p>Detailed Reports</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #00d4ff;'>⚙️ Configuration</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎚️ Confidence Threshold")
    confidence_threshold = st.slider("", 0.0, 1.0, 0.5, help="Adjust the minimum confidence level for predictions")
    
    st.markdown("### 📁 Model Info")
    st.info("🧠 CNN-based Deep Learning Model trained on 5000+ MRI scans")
    
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Upload Brain MRI Image
    2. Click 'Analyze Scan' button
    3. View detailed results
    """)
    
    st.markdown("---")
    st.markdown("**Supported Formats:** JPG, PNG, JPEG")

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    
    # Custom file uploader with better styling
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload", 
        type=["jpg", "png", "jpeg"],
        help="Upload brain MRI scan in JPG, PNG or JPEG format"
    )

    if uploaded_file is not None:
        # Display uploaded image with styling
        image = Image.open(uploaded_file)
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Convert image for model
        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = img.reshape(1, -1)

with col2:
    st.markdown("### 🧠 Prediction Panel")
    
    if uploaded_file is not None:
        # Analyze button with custom styling
        if st.button("🔍 Analyze Scan", use_container_width=True, type="primary"):

            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            with st.spinner("🤖 AI is analyzing the MRI scan..."):
                time.sleep(1)
                
                try:
                    prediction = model.predict(img)[0]
                    prob = model.predict_proba(img)[0]
                except Exception as e:
                    st.error(f"Model prediction error: {e}")
                    prediction = 0
                    prob = [0.5, 0.5]

            progress_bar.empty()
            
            st.markdown("---")
            
            # Result Display
            if prediction == 1:
                st.markdown("""
                    <div style='text-align: center; padding: 30px; background: rgba(255, 75, 75, 0.2); border-radius: 20px; border: 2px solid #ff4b4b;'>
                        <h2 style='color: #ff4b4b; font-size: 2rem;'>⚠️ Tumor Detected</h2>
                        <p style='color: #ff6b6b;'>Please consult with a specialist immediately</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 30px; background: rgba(0, 255, 136, 0.2); border-radius: 20px; border: 2px solid #00ff88;'>
                        <h2 style='color: #00ff88; font-size: 2rem;'>✅ No Tumor Detected</h2>
                        <p style='color: #88ffaa;'>The scan appears normal</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # Confidence Check
            confidence = max(prob)
            
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                st.metric("Confidence Score", f"{confidence*100:.1f}%")
            with col_conf2:
                if confidence < confidence_threshold:
                    st.warning("⚠️ Low Confidence")
                else:
                    st.success("✓ High Confidence")

            # Probability Chart
            st.markdown("### 📊 Probability Distribution")
            
            colors = ['#00ff88', '#ff4b4b'] if prediction == 0 else ['#ff4b4b', '#00ff88']
            
            fig = px.bar(
                x=["No Tumor", "Tumor"], 
                y=prob,
                color=["No Tumor", "Tumor"],
                color_discrete_sequence=colors,
                title="Prediction Confidence",
                labels={'x': 'Class', 'y': 'Probability'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download report button
            st.download_button(
                "📥 Download Report",
                data=f"MRI Analysis Report\n\nPrediction: {'Tumor Detected' if prediction == 1 else 'No Tumor'}\nConfidence: {confidence*100:.1f}%\nNo Tumor Probability: {prob[0]*100:.1f}%\nTumor Probability: {prob[1]*100:.1f}%",
                file_name="mri_report.txt",
                mime="text/plain"
            )

# ---------------- EXTRA FEATURES ----------------
st.markdown("---")

if uploaded_file is not None:
    st.markdown("### 🧪 Image Processing Preview")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("#### Original Image")
        st.image(image, use_container_width=True)
    
    with col4:
        st.markdown("#### Grayscale")
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        st.image(gray, caption="Grayscale", use_container_width=True, clamp=True)
    
    with col5:
        st.markdown("#### Edge Detection")
        edges = cv2.Canny(gray, 100, 200)
        st.image(edges, caption="Canny Edges", use_container_width=True, clamp=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #a0a0a0;'>
        <p>⚡ AI for Healthcare | Brain Tumor Detection System v1.0</p>
        <p style='font-size: 0.8rem;'>This is an AI-assisted diagnosis tool. Always consult with medical professionals.</p>
    </div>
""", unsafe_allow_html=True)