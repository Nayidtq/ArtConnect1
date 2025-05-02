import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from transformers import pipeline
from art_style_classifier import ArtStyleClassifier, get_style_prediction
from market_analysis import MarketAnalyzer

# Page configuration
st.set_page_config(
    page_title="ArtConnect - Artwork Analysis",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 ArtConnect - Artwork Analysis")
st.markdown("""
This application uses computer vision to analyze artworks and provide:
- Artistic style
- Potential customers
- Price estimation
- Marketing strategies
""")

# Load models and analyzers
@st.cache_resource
def load_models():
    # Style classification model
    style_model = ArtStyleClassifier()
    style_model.eval()
    
    # Text analysis pipeline
    text_analyzer = pipeline("text-generation", model="gpt2")
    
    # Market analyzer
    market_analyzer = MarketAnalyzer()
    
    return style_model, text_analyzer, market_analyzer

# Image processing function
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main interface
def main():
    style_model, text_analyzer, market_analyzer = load_models()
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded artwork", use_column_width=True)
        
        # Process image
        if st.button("Analyze Artwork"):
            with st.spinner("Analyzing artwork..."):
                # Preprocess image
                input_tensor = process_image(image)
                
                # Get style prediction
                predicted_style, confidence = get_style_prediction(style_model, input_tensor)
                
                # Get market analysis
                price_range = market_analyzer.get_price_range(predicted_style)
                customer_profile = market_analyzer.get_customer_profile(predicted_style)
                marketing_strategy = market_analyzer.get_marketing_strategy(predicted_style)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Artistic style
                st.markdown("### 🎨 Artistic Style")
                st.write(f"Based on visual analysis, this artwork appears to belong to the style: **{predicted_style}**")
                st.write(f"Prediction confidence: {confidence:.2%}")
                
                # Potential customers
                st.markdown("### 👥 Potential Customers")
                st.write(f"Potential customer profile: {customer_profile}")
                
                # Price estimation
                st.markdown("### 💰 Price Estimation")
                st.write(f"Suggested price range: ${price_range[0]:,} - ${price_range[1]:,}")
                
                # Marketing strategy
                st.markdown("### 📈 Marketing Strategy")
                st.write("Marketing recommendations:")
                st.write(marketing_strategy)

if __name__ == "__main__":
    main() 