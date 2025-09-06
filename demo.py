#!/usr/bin/env python3
"""
People Detection Demo
====================

A demonstration script showing how to use the people detection model.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os

def demo_people_detection():
    """Demonstrate the people detection model"""
    print("🎯 People Detection Model Demo")
    print("=" * 40)
    
    # Load the model
    model_path = "people_detection_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please make sure the model file exists in the current directory.")
        return
    
    try:
        print("Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with a sample image from the test dataset
    test_dir = "data/test"
    if os.path.exists(test_dir):
        # Find the first image in the test directory
        image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            sample_image = os.path.join(test_dir, image_files[0])
            print(f"\n🖼️  Testing with sample image: {image_files[0]}")
            
            # Load and preprocess the image
            img = Image.open(sample_image).convert('RGB').resize((224, 224))
            img_array = np.array(img).astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = model.predict(img_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # Display results
            if predicted_class == 1:
                result = "Person detected"
                emoji = "✅"
            else:
                result = "No person detected"
                emoji = "❌"
            
            print(f"{emoji} Result: {result}")
            print(f"📊 Confidence: {confidence:.2f}")
            print(f"🎯 Raw predictions: {predictions[0]}")
            
            # Show how to use the model programmatically
            print(f"\n💡 How to use this model in your code:")
            print(f"```python")
            print(f"import tensorflow as tf")
            print(f"import numpy as np")
            print(f"from PIL import Image")
            print(f"")
            print(f"# Load model")
            print(f"model = tf.keras.models.load_model('{model_path}')")
            print(f"")
            print(f"# Load and preprocess image")
            print(f"img = Image.open('your_image.jpg').convert('RGB').resize((224, 224))")
            print(f"img_array = np.array(img).astype(np.float32) / 255.0")
            print(f"img_batch = np.expand_dims(img_array, axis=0)")
            print(f"")
            print(f"# Make prediction")
            print(f"predictions = model.predict(img_batch, verbose=0)")
            print(f"predicted_class = np.argmax(predictions[0])")
            print(f"confidence = float(np.max(predictions[0]))")
            print(f"")
            print(f"# Interpret results")
            print(f"if predicted_class == 1:")
            print(f"    result = 'Person detected'")
            print(f"else:")
            print(f"    result = 'No person detected'")
            print(f"```")
            
        else:
            print(f"❌ No images found in {test_dir}")
    else:
        print(f"❌ Test directory not found: {test_dir}")
    
    print(f"\n🚀 Ready to detect people in your images!")
    print(f"Use the test scripts to try it with your own images:")

if __name__ == "__main__":
    demo_people_detection()
