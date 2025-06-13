#!/usr/bin/env python3
"""
🖼️ CIFAR-10 Image Classification Web App

A professional Streamlit interface for image classification using trained CNN models.
Upload an image and get real-time predictions with confidence scores!

Author: AI Assistant
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io
import base64
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from src.models.cifar10_cnn import create_cifar10_model
    from src.data.transforms import ImageTransforms
except ImportError:
    st.error("⚠️ Please ensure you're running from the project root directory")
    st.stop()

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/cv-image-classifier',
        'Report a bug': 'https://github.com/your-repo/cv-image-classifier/issues',
        'About': 'CIFAR-10 Image Classification with PyTorch & Streamlit'
    }
)

# ===== CONSTANTS =====
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_DESCRIPTIONS = {
    'airplane': '✈️ Aircraft including planes, jets, and helicopters',
    'automobile': '🚗 Cars, sedans, and passenger vehicles',
    'bird': '🐦 Various bird species in flight or perched',
    'cat': '🐱 Domestic cats in different poses',
    'deer': '🦌 Deer and similar wildlife animals',
    'dog': '🐕 Dogs of various breeds and sizes',
    'frog': '🐸 Frogs, toads, and amphibians',
    'horse': '🐎 Horses and equine animals',
    'ship': '🚢 Ships, boats, and watercraft',
    'truck': '🚛 Trucks, lorries, and commercial vehicles'
}

# ===== STYLING =====
def load_css():
    """Load custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== CACHING FUNCTIONS =====
@st.cache_resource
def load_model(model_path, model_type='simple'):
    """Load the trained model with caching"""
    try:
        if not os.path.exists(model_path):
            return None, "Model file not found"
        
        # Create model architecture
        model = create_cifar10_model(model_type, num_classes=10)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data
def get_image_transforms():
    """Get image preprocessing transforms"""
    return ImageTransforms.get_test_transforms()

@st.cache_data
def create_sample_images():
    """Create sample images for testing"""
    # This would ideally load actual sample images from CIFAR-10
    # For now, return placeholder information
    samples = {
        'airplane': 'Sample airplane image',
        'car': 'Sample car image',
        'bird': 'Sample bird image',
        'cat': 'Sample cat image',
        'dog': 'Sample dog image'
    }
    return samples

# ===== HELPER FUNCTIONS =====
def preprocess_image(uploaded_file):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert to PIL Image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Get transforms
        transform = get_image_transforms()
        
        # Apply transforms
        image_tensor = transform(image)
        
        return image, image_tensor, None
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}"

def make_prediction(model, image_tensor, top_k=5):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor.unsqueeze(0))
            
            # Get probabilities
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(CIFAR10_CLASSES)))
            
            # Format results
            predictions = []
            for i in range(len(top_indices[0])):
                class_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                class_name = CIFAR10_CLASSES[class_idx]
                
                predictions.append({
                    'class': class_name,
                    'confidence': confidence * 100,
                    'description': CLASS_DESCRIPTIONS[class_name]
                })
            
            return predictions, None
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence >= 70:
        return "confidence-high"
    elif confidence >= 40:
        return "confidence-medium"
    else:
        return "confidence-low"

def create_confidence_chart(predictions):
    """Create interactive confidence chart"""
    df = pd.DataFrame(predictions)
    
    # Create bar chart
    fig = px.bar(
        df, 
        x='confidence', 
        y='class',
        orientation='h',
        title='Prediction Confidence Scores',
        color='confidence',
        color_continuous_scale='Viridis',
        text='confidence'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="Class",
        font=dict(size=12)
    )
    
    return fig

# ===== MAIN APP COMPONENTS =====
def render_header():
    """Render the main header section"""
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ CIFAR-10 Image Classifier</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">
            Upload an image and get real-time AI predictions with confidence scores!
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with model information and settings"""
    with st.sidebar:
        st.markdown("## 🎛️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["simple", "advanced", "resnet"],
            index=0,
            help="Choose the CNN architecture for classification"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=5.0,
            help="Minimum confidence to display predictions"
        )
        
        # Number of predictions to show
        top_k = st.slider(
            "Top K Predictions",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of top predictions to display"
        )
        
        st.markdown("---")
        
        # Model information
        st.markdown("## 📊 Model Info")
        
        model_info = {
            "simple": {"params": "95K", "desc": "3-layer CNN"},
            "advanced": {"params": "2.8M", "desc": "ResNet blocks"},
            "resnet": {"params": "272K", "desc": "CIFAR-10 optimized"}
        }
        
        info = model_info.get(model_type, model_info["simple"])
        st.metric("Parameters", info["params"])
        st.write(f"**Architecture:** {info['desc']}")
        
        st.markdown("---")
        
        # Class information
        st.markdown("## 🏷️ CIFAR-10 Classes")
        with st.expander("View all classes"):
            for class_name in CIFAR10_CLASSES:
                st.write(f"• {CLASS_DESCRIPTIONS[class_name]}")
        
        return model_type, confidence_threshold, top_k

def render_upload_section():
    """Render the image upload section"""
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a PNG, JPG, or JPEG image (max 200MB)",
        label_visibility="collapsed"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-section">
            <h3>📁 Drag and drop an image here</h3>
            <p>or click to browse your files</p>
            <p><em>Supported formats: PNG, JPG, JPEG</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    return uploaded_file

def render_sample_images():
    """Render sample images section"""
    st.markdown("### 🖼️ Sample Images")
    
    with st.expander("Try with sample images"):
        st.info("Sample images would be displayed here. You can download CIFAR-10 test images and place them in a 'samples' folder.")
        
        # Placeholder for sample images
        cols = st.columns(5)
        sample_classes = ['airplane', 'car', 'bird', 'cat', 'dog']
        
        for i, class_name in enumerate(sample_classes):
            with cols[i]:
                st.write(f"**{class_name.title()}**")
                st.write("Sample image placeholder")
                if st.button(f"Use {class_name}", key=f"sample_{class_name}"):
                    st.info(f"Sample {class_name} image would be loaded here")

def render_results(predictions, image):
    """Render prediction results"""
    if not predictions:
        return
    
    st.markdown("### 🔍 Prediction Results")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Top prediction highlight
        top_pred = predictions[0]
        confidence_class = get_confidence_color(top_pred['confidence'])
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🎯 Top Prediction</h3>
            <h2>{top_pred['class'].title()}</h2>
            <h3 class="{confidence_class}">{top_pred['confidence']:.1f}%</h3>
            <p>{top_pred['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence chart
        fig = create_confidence_chart(predictions)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed predictions table
    st.markdown("#### 📊 Detailed Predictions")
    
    df = pd.DataFrame(predictions)
    df['confidence'] = df['confidence'].round(2)
    df.columns = ['Class', 'Confidence (%)', 'Description']
    
    # Style the dataframe
    def highlight_top(row):
        if row.name == 0:  # Top prediction
            return ['background-color: #d4edda'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = df.style.apply(highlight_top, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def render_error_message(error):
    """Render error message"""
    st.error(f"❌ {error}")
    
    with st.expander("Troubleshooting Tips"):
        st.markdown("""
        **Common issues:**
        - Make sure the image is in PNG, JPG, or JPEG format
        - Ensure the image file is not corrupted
        - Try with a smaller image size (< 10MB)
        - Check that the model file exists in the checkpoints folder
        
        **Need help?** Check the documentation or report an issue.
        """)

# ===== MAIN APP =====
def main():
    """Main application function"""
    # Load custom CSS
    load_css()
    
    # Render header
    render_header()
    
    # Render sidebar and get settings
    model_type, confidence_threshold, top_k = render_sidebar()
    
    # Try to load model
    model_path = f"checkpoints/cifar10_simple_cnn/model_best.pth"
    model, model_error = load_model(model_path, model_type)
    
    if model_error:
        st.error(f"❌ {model_error}")
        st.info("Please ensure you have trained a model first by running the training script.")
        st.code("python src/training/train_cifar10_clean.py")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Classify", "🖼️ Sample Images", "ℹ️ About"])
    
    with tab1:
        # Upload section
        uploaded_file = render_upload_section()
        
        if uploaded_file is not None:
            # Process the uploaded image
            with st.spinner("🔄 Processing image..."):
                image, image_tensor, error = preprocess_image(uploaded_file)
            
            if error:
                render_error_message(error)
                return
            
            # Make prediction
            with st.spinner("🧠 Making prediction..."):
                predictions, pred_error = make_prediction(model, image_tensor, top_k)
            
            if pred_error:
                render_error_message(pred_error)
                return
            
            # Filter predictions by confidence threshold
            filtered_predictions = [
                pred for pred in predictions 
                if pred['confidence'] >= confidence_threshold
            ]
            
            if not filtered_predictions:
                st.warning(f"⚠️ No predictions above {confidence_threshold}% confidence threshold.")
                st.info("Try lowering the confidence threshold in the sidebar.")
                filtered_predictions = predictions[:3]  # Show top 3 anyway
            
            # Render results
            render_results(filtered_predictions, image)
    
    with tab2:
        render_sample_images()
    
    with tab3:
        st.markdown("""
        ## 🧠 About This App
        
        This image classification app uses a **Convolutional Neural Network (CNN)** trained on the CIFAR-10 dataset.
        
        ### 📚 Model Details
        - **Dataset**: CIFAR-10 (60,000 32×32 color images)
        - **Classes**: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        - **Architecture**: Simple CNN with 3 convolutional layers
        - **Training**: 10 epochs, 68% validation accuracy
        
        ### 🔧 How It Works
        1. **Upload**: Choose an image file (PNG, JPG, JPEG)
        2. **Preprocess**: Image is resized to 32×32 and normalized
        3. **Inference**: CNN model predicts the class probabilities
        4. **Results**: Top predictions with confidence scores are displayed
        
        ### 📊 Performance Notes
        - Model works best with clear, centered objects
        - Optimized for the 10 CIFAR-10 classes
        - Real-world images may have different characteristics than training data
        
        ### 🚀 Technology Stack
        - **Framework**: PyTorch
        - **Web Interface**: Streamlit
        - **Visualization**: Plotly
        - **Image Processing**: PIL, torchvision
        """)

if __name__ == "__main__":
    main()