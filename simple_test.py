#!/usr/bin/env python3
"""
Simple People Detection Test
============================

A simple script to test the people detection model with a single image or batch of images.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

def load_model(model_path="people_detection_model.h5"):
    """Load the trained model"""
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_image(model, image_path):
    """Predict if an image contains a person"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Interpret results
        if predicted_class == 1:
            result = "Person detected"
        else:
            result = "No person detected"
        
        return result, confidence
    
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None, None

def test_single_image(image_path):
    """Test a single image"""
    model = load_model()
    if model is None:
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🖼️  Testing image: {os.path.basename(image_path)}")
    result, confidence = predict_image(model, image_path)
    
    if result:
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}")
        
        # Visual indicator
        if "Person detected" in result:
            print("✅ Person found in image!")
        else:
            print("❌ No person found in image")

def test_sample_images():
    """Test with sample images from the test dataset"""
    model = load_model()
    if model is None:
        return
    
    test_dir = "data/test"
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    # Get first 5 images from test directory
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    if not image_files:
        print("❌ No images found in test directory")
        return
    
    print(f"\n🧪 Testing {len(image_files)} sample images from test dataset:")
    print("-" * 60)
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(test_dir, filename)
        result, confidence = predict_image(model, image_path)
        
        if result:
            status = "✅" if "Person detected" in result else "❌"
            print(f"{status} {i}. {filename:<40} | {result:<20} | {confidence:.2f}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Test specific image
        image_path = sys.argv[1]
        test_single_image(image_path)
    else:
        # Test sample images
        test_sample_images()

if __name__ == "__main__":
    main()
