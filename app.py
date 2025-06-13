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

sys.path.append(os.path.dirname(__file__))

from src.models.models import create_model
from src.data.transforms import ImageTransforms
from src.utils.utils import load_checkpoint

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

@st.cache_resource
def load_model(checkpoint_path, model_name):
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        num_classes = len(class_to_idx)
        model = create_model(model_name, num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        transform = ImageTransforms.get_test_transforms()
        
        return model, idx_to_class, transform, class_to_idx
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def predict_image(model, image, transform, idx_to_class, top_k=5):
    if model is None:
        return None
    
    try:
        # Transform image
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(idx_to_class)))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        predictions = []
        for i in range(len(top_indices)):
            class_name = idx_to_class[top_indices[i]]
            confidence = top_probs[i]
            predictions.append({
                'Class': class_name,
                'Confidence': float(confidence)
            })
        
        return predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title("🖼️ Image Classification System")
    st.markdown("Upload an image to classify it using a trained PyTorch model.")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Check for available checkpoints
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            selected_checkpoint = st.sidebar.selectbox(
                "Select Model Checkpoint",
                checkpoint_files
            )
            checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
        else:
            st.sidebar.warning("No checkpoint files found in checkpoints/ directory")
            checkpoint_path = None
    else:
        st.sidebar.warning("Checkpoints directory not found")
        checkpoint_path = None
    
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        ["resnet18", "resnet50", "efficientnet_b0", "custom_cnn"]
    )
    
    top_k = st.sidebar.slider("Top K Predictions", 1, 10, 5)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG, JPG, or JPEG image"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Load model and make prediction
            if checkpoint_path:
                with st.spinner("Loading model..."):
                    model, idx_to_class, transform, class_to_idx = load_model(checkpoint_path, model_name)
                
                if model is not None:
                    with st.spinner("Making prediction..."):
                        predictions = predict_image(model, image, transform, idx_to_class, top_k)
                    
                    if predictions:
                        with col2:
                            st.header("Predictions")
                            
                            # Create DataFrame for better display
                            df = pd.DataFrame(predictions)
                            df['Confidence'] = df['Confidence'].round(4)
                            df['Confidence %'] = (df['Confidence'] * 100).round(2)
                            
                            # Display top prediction prominently
                            st.success(f"**Top Prediction:** {df.iloc[0]['Class']} ({df.iloc[0]['Confidence %']:.2f}%)")
                            
                            # Display all predictions in a table
                            st.subheader("All Predictions")
                            st.dataframe(df[['Class', 'Confidence %']], use_container_width=True)
                            
                            # Create bar chart
                            fig = px.bar(
                                df, 
                                x='Confidence %', 
                                y='Class',
                                orientation='h',
                                title='Prediction Confidence',
                                color='Confidence %',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create pie chart for top 5 predictions
                            if len(df) > 1:
                                fig_pie = px.pie(
                                    df.head(5), 
                                    values='Confidence %', 
                                    names='Class',
                                    title='Top 5 Predictions Distribution'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.error("Please ensure you have a trained model checkpoint in the checkpoints/ directory")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.subheader("How to Use")
        st.markdown("""
        1. Select a model checkpoint from the sidebar
        2. Choose the model architecture
        3. Upload an image using the file uploader
        4. View the predictions and confidence scores
        """)
    
    with info_col2:
        st.subheader("Supported Formats")
        st.markdown("""
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - Maximum file size: 200MB
        - Recommended: Square images work best
        """)
    
    with info_col3:
        st.subheader("Model Info")
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'class_to_idx' in checkpoint:
                    num_classes = len(checkpoint['class_to_idx'])
                    st.write(f"Number of classes: {num_classes}")
                    if 'best_acc' in checkpoint:
                        st.write(f"Best accuracy: {checkpoint['best_acc']:.2f}%")
            except:
                st.write("Could not load model info")
        else:
            st.write("No model loaded")

if __name__ == "__main__":
    main()