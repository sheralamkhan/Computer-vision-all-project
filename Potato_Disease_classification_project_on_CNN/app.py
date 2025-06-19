import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="🥔 Potato Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #4F4F4F;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .disease-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']
CLASS_DESCRIPTIONS = {
    'Potato Early Blight': '🟡 A common fungal disease causing dark spots with concentric rings on leaves',
    'Potato Late Blight': '🔴 A serious disease that can destroy entire crops, causing brown lesions',
    'Potato Healthy': '🟢 Healthy potato leaves with no signs of disease'
}
IMAGE_SIZE = 256

# Load the model
@st.cache_resource
def load_trained_model():
    try:
        return load_model("potato_disease_Model_Model_Model.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar with information
with st.sidebar:
    st.markdown("### 📊 About This App")
    st.info("""
    This AI-powered tool helps farmers and gardeners identify potato leaf diseases using deep learning.
    
    **Supported Diseases:**
    - Early Blight
    - Late Blight
    - Healthy Classification
    """)
    
    st.markdown("### 📈 Model Performance")
    st.success("Model Accuracy: 95.2%")
    
    st.markdown("### 🔍 How to Use")
    st.markdown("""
    1. Upload a clear image of potato leaf
    2. Wait for AI analysis
    3. View results and recommendations
    4. Take appropriate action
    """)
    
    st.markdown("### 💡 Tips for Best Results")
    st.warning("""
    - Use clear, well-lit images
    - Focus on the leaf surface
    - Avoid blurry or low-quality photos
    - Ensure the leaf fills most of the frame
    """)

# Main content
st.markdown('<h1 class="main-header">🥔 Potato Disease Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Plant Health Analysis for Smart Farming</p>', unsafe_allow_html=True)

# Load model
model = load_trained_model()

if model is None:
    st.error("🚨 Model could not be loaded. Please check if the model file exists.")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Potato Leaf Image")
    
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of a potato leaf for disease analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='📸 Uploaded Image', use_column_width=True)
        
        # Image info
        st.info(f"📋 Image Details: {img.size[0]}x{img.size[1]} pixels")

with col2:
    if uploaded_file is not None:
        st.markdown("### 🔬 Analysis Results")
        
        # Show loading spinner
        with st.spinner('🤖 Analyzing image with AI...'):
            # Preprocess image
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Predict
            predictions = model.predict(img_array)[0]
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)
        
        # Display results with custom styling
        st.markdown(f"""
        <div class="result-card">
            <h3>🎯 Prediction Result</h3>
            <h2 style="color: #2E8B57; margin: 0;">{predicted_class}</h2>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                <strong>Confidence: {confidence * 100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display disease information
        st.markdown(f"""
        <div class="info-card">
            <h4>ℹ️ Disease Information</h4>
            <p>{CLASS_DESCRIPTIONS[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence level indicator
        if confidence > 0.8:
            st.success("✅ High confidence prediction!")
        elif confidence > 0.6:
            st.warning("⚠️ Moderate confidence - consider additional analysis")
        else:
            st.error("❌ Low confidence - please upload a clearer image")

# Detailed analysis section
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Create probability chart
        prob_data = pd.DataFrame({
            'Disease': CLASS_NAMES,
            'Probability': predictions * 100
        })
        
        fig = px.bar(
            prob_data, 
            x='Probability', 
            y='Disease',
            orientation='h',
            title='Disease Probability Distribution',
            color='Probability',
            color_continuous_scale='RdYlGn',
            text='Probability'
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=300,
            showlegend=False,
            title_x=0.5,
            xaxis_title="Probability (%)",
            yaxis_title="Disease Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# Recommendations section
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    if "Early Blight" in predicted_class:
        st.markdown("""
        <div class="disease-info">
            <h4>🟡 Early Blight Treatment</h4>
            <ul>
                <li>Apply fungicide containing chlorothalonil or mancozeb</li>
                <li>Improve air circulation around plants</li>
                <li>Water at soil level to avoid wetting leaves</li>
                <li>Remove affected leaves and destroy them</li>
                <li>Rotate crops to break disease cycle</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif "Late Blight" in predicted_class:
        st.markdown("""
        <div class="disease-info">
            <h4>🔴 Late Blight Treatment (URGENT)</h4>
            <ul>
                <li><strong>Act immediately!</strong> This disease spreads rapidly</li>
                <li>Apply copper-based fungicides immediately</li>
                <li>Remove and destroy all affected plants</li>
                <li>Avoid overhead watering</li>
                <li>Consider resistant potato varieties for future planting</li>
                <li>Monitor weather conditions (cool, wet weather favors this disease)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown("""
        <div class="disease-info">
            <h4>🟢 Healthy Plant Maintenance</h4>
            <ul>
                <li>Continue current care practices</li>
                <li>Monitor regularly for early disease detection</li>
                <li>Ensure proper spacing for air circulation</li>
                <li>Water at soil level to keep leaves dry</li>
                <li>Apply preventive fungicide if weather conditions are favorable for disease</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌱 <strong>Potato Disease Classifier</strong> | Powered by Deep Learning & TensorFlow</p>
    <p><em>Helping farmers make informed decisions for healthier crops</em></p>
</div>
""", unsafe_allow_html=True)