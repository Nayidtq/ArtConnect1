import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from transformers import pipeline
from art_style_classifier import ArtStyleClassifier, get_style_prediction
from market_analysis import MarketAnalyzer
import time

# Page configuration
st.set_page_config(
    page_title="ArtConnect - Artwork Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a sidebar for additional information
with st.sidebar:
    st.title("About ArtConnect")
    st.markdown("""
    ArtConnect uses advanced computer vision and machine learning to analyze artworks and provide:
    - 🎨 Style classification
    - 👥 Market analysis
    - 💰 Price estimation
    - 📈 Marketing strategies
    """)
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload an artwork image (JPG, PNG)
    2. Click 'Analyze Artwork'
    3. View detailed analysis results
    """)

# Title and description
st.title("🎨 ArtConnect - Artwork Analysis")
st.markdown("""
This application uses computer vision to analyze artworks and provide:
- Artistic style
- Potential customers
- Price estimation
- Marketing strategies
""")

# Add a file uploader with size limit
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def validate_image(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File size too large. Please upload an image smaller than 5MB.")
        return False
    return True

# Load models and analyzers
@st.cache_resource
def load_models():
    try:
        # Style classification model
        style_model = ArtStyleClassifier()
        style_model.eval()
        
        # Text analysis pipeline
        text_analyzer = pipeline("text-generation", model="gpt2")
        
        # Market analyzer
        market_analyzer = MarketAnalyzer()
        
        return style_model, text_analyzer, market_analyzer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Image processing function
def process_image(image):
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Main interface
def main():
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Load models
    with st.spinner("Loading models..."):
        style_model, text_analyzer, market_analyzer = load_models()
    
    if style_model is None:
        st.error("Failed to load required models. Please try again later.")
        return
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if not validate_image(uploaded_file):
            return
            
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded artwork", use_column_width=True)
            
            # Process image
            if st.button("Analyze Artwork"):
                with st.spinner("Analyzing artwork..."):
                    # Preprocess image
                    input_tensor = process_image(image)
                    if input_tensor is None:
                        return
                    
                    # Get style prediction
                    predicted_style, confidence = get_style_prediction(style_model, input_tensor)
                    
                    # Get market analysis
                    price_range = market_analyzer.get_price_range(predicted_style)
                    customer_profile = market_analyzer.get_customer_profile(predicted_style)
                    marketing_strategy = market_analyzer.get_marketing_strategy(predicted_style)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Artistic style
                        st.markdown("### 🎨 Artistic Style")
                        st.write(f"Based on visual analysis, this artwork appears to belong to the style: **{predicted_style}**")
                        st.write(f"Prediction confidence: {confidence:.2%}")
                        
                        # Price estimation
                        st.markdown("### 💰 Price Estimation")
                        st.write(f"Suggested price range: ${price_range[0]:,} - ${price_range[1]:,}")
                    
                    with col2:
                        # Potential customers
                        st.markdown("### 👥 Potential Customers")
                        st.write(f"Potential customer profile: {customer_profile}")
                        
                        # Marketing strategy
                        st.markdown("### 📈 Marketing Strategy")
                        st.write("Marketing recommendations:")
                        st.write(marketing_strategy)
                    
                    st.session_state.analysis_complete = True
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with a different image.")

if __name__ == "__main__":
    main() 