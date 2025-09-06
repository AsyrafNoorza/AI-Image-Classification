#!/usr/bin/env python3
"""
People Detection Model Tester
============================

This script provides various ways to test the people detection model:
1. Test with a single image
2. Test with all images in a directory
3. Test with images from the test dataset
4. Batch testing with performance metrics
5. Interactive testing with image display

Usage:
    python test_model.py --help
    python test_model.py --image path/to/image.jpg
    python test_model.py --directory path/to/images/
    python test_model.py --test-dataset
    python test_model.py --batch-test
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path
import glob

class PeopleDetectionTester:
    def __init__(self, model_path="people_detection_model.h5"):
        """Initialize the tester with the trained model"""
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure the model file exists and is compatible with your TensorFlow version.")
            exit(1)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        try:
            # Load and resize image
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize(self.img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized)
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            return img_batch, img
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None, None
    
    def predict_single_image(self, image_path, show_image=False):
        """Predict if a single image contains a person"""
        img_batch, original_img = self.preprocess_image(image_path)
        if img_batch is None:
            return None, None, None
        
        # Make prediction
        start_time = time.time()
        predictions = self.model.predict(img_batch, verbose=0)
        prediction_time = time.time() - start_time
        
        # Get results
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Interpret results
        if predicted_class == 1:
            result = "Person detected"
        else:
            result = "No person detected"
        
        # Display image if requested
        if show_image:
            plt.figure(figsize=(8, 6))
            plt.imshow(original_img)
            plt.title(f"{result}\nConfidence: {confidence:.2f}\nTime: {prediction_time:.3f}s")
            plt.axis('off')
            plt.show()
        
        return result, confidence, prediction_time
    
    def test_directory(self, directory_path, show_results=True):
        """Test all images in a directory"""
        if not os.path.exists(directory_path):
            print(f"❌ Directory {directory_path} does not exist!")
            return
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(directory_path, f"*{ext.upper()}")))
        
        if not image_files:
            print(f"❌ No image files found in {directory_path}")
            return
        
        print(f"🔍 Testing {len(image_files)} images in {directory_path}")
        print("-" * 60)
        
        results = []
        total_time = 0
        
        for i, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            result, confidence, pred_time = self.predict_single_image(image_path)
            
            if result is not None:
                results.append({
                    'filename': filename,
                    'result': result,
                    'confidence': confidence,
                    'time': pred_time
                })
                total_time += pred_time
                
                if show_results:
                    status = "✅" if "Person detected" in result else "❌"
                    print(f"{status} {i:3d}/{len(image_files)} | {filename:<40} | {result:<20} | {confidence:.2f} | {pred_time:.3f}s")
        
        # Summary
        if results:
            person_detected = sum(1 for r in results if "Person detected" in r['result'])
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_time = total_time / len(results)
            
            print("-" * 60)
            print(f"📊 SUMMARY:")
            print(f"   Total images: {len(results)}")
            print(f"   Person detected: {person_detected}")
            print(f"   No person detected: {len(results) - person_detected}")
            print(f"   Average confidence: {avg_confidence:.2f}")
            print(f"   Average prediction time: {avg_time:.3f}s")
            print(f"   Total time: {total_time:.3f}s")
        
        return results
    
    def test_dataset(self, test_dir="data/test"):
        """Test with the provided test dataset"""
        if not os.path.exists(test_dir):
            print(f"❌ Test directory {test_dir} does not exist!")
            return
        
        print(f"🧪 Testing with dataset in {test_dir}")
        return self.test_directory(test_dir, show_results=True)
    
    def batch_test_with_metrics(self, test_dir="data/test"):
        """Perform batch testing with detailed metrics"""
        results = self.test_dataset(test_dir)
        if not results:
            return
        
        # Calculate detailed metrics
        person_detected = [r for r in results if "Person detected" in r['result']]
        no_person = [r for r in results if "No person detected" in r['result']]
        
        print("\n📈 DETAILED METRICS:")
        print("=" * 50)
        
        if person_detected:
            person_confidences = [r['confidence'] for r in person_detected]
            print(f"Person Detection:")
            print(f"  Count: {len(person_detected)}")
            print(f"  Avg Confidence: {np.mean(person_confidences):.3f}")
            print(f"  Min Confidence: {np.min(person_confidences):.3f}")
            print(f"  Max Confidence: {np.max(person_confidences):.3f}")
            print(f"  Std Deviation: {np.std(person_confidences):.3f}")
        
        if no_person:
            no_person_confidences = [r['confidence'] for r in no_person]
            print(f"\nNo Person Detection:")
            print(f"  Count: {len(no_person)}")
            print(f"  Avg Confidence: {np.mean(no_person_confidences):.3f}")
            print(f"  Min Confidence: {np.min(no_person_confidences):.3f}")
            print(f"  Max Confidence: {np.max(no_person_confidences):.3f}")
            print(f"  Std Deviation: {np.std(no_person_confidences):.3f}")
        
        # Performance metrics
        all_times = [r['time'] for r in results]
        print(f"\nPerformance:")
        print(f"  Total Images: {len(results)}")
        print(f"  Total Time: {sum(all_times):.3f}s")
        print(f"  Average Time per Image: {np.mean(all_times):.3f}s")
        print(f"  Images per Second: {len(results)/sum(all_times):.2f}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("🎮 Interactive Testing Mode")
        print("Enter image paths to test (or 'quit' to exit)")
        print("-" * 40)
        
        while True:
            image_path = input("\nEnter image path: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            result, confidence, pred_time = self.predict_single_image(image_path, show_image=True)
            if result:
                print(f"Result: {result}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Prediction time: {pred_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description="Test the People Detection Model")
    parser.add_argument("--model", default="people_detection_model.h5", 
                       help="Path to the model file")
    parser.add_argument("--image", help="Test a single image")
    parser.add_argument("--directory", help="Test all images in a directory")
    parser.add_argument("--test-dataset", action="store_true", 
                       help="Test with the provided test dataset")
    parser.add_argument("--batch-test", action="store_true", 
                       help="Perform batch testing with detailed metrics")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive testing mode")
    parser.add_argument("--show-image", action="store_true", 
                       help="Show image when testing single image")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PeopleDetectionTester(args.model)
    
    # Execute based on arguments
    if args.image:
        print(f"🖼️  Testing single image: {args.image}")
        result, confidence, pred_time = tester.predict_single_image(args.image, args.show_image)
        if result:
            print(f"Result: {result}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Prediction time: {pred_time:.3f}s")
    
    elif args.directory:
        tester.test_directory(args.directory)
    
    elif args.test_dataset:
        tester.test_dataset()
    
    elif args.batch_test:
        tester.batch_test_with_metrics()
    
    elif args.interactive:
        tester.interactive_test()
    
    else:
        # Default: test with sample images from test dataset
        print("🔍 Running default test with sample images...")
        tester.test_dataset()

if __name__ == "__main__":
    main()
