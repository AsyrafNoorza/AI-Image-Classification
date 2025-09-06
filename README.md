# People Detection Model Testing

This directory contains several scripts to test your people detection model. Here's how to use them:

## 🚀 Quick Start

### 1. Demo Script (Recommended for first-time users)
```bash
python demo.py
```
This will show you how the model works with a sample image and provide code examples.

### 2. Simple Test
```bash
# Test with sample images from the dataset
python simple_test.py

# Test with your own image
python simple_test.py path/to/your/image.jpg
```

### 3. Advanced Testing
```bash
# Test a single image
python test_model.py --image path/to/image.jpg

# Test all images in a directory
python test_model.py --directory path/to/images/

# Test with the provided test dataset
python test_model.py --test-dataset

# Batch testing with detailed metrics
python test_model.py --batch-test

# Interactive testing mode
python test_model.py --interactive
```

## 📁 Files Created

- **`demo.py`** - Simple demonstration of the model
- **`simple_test.py`** - Basic testing script
- **`test_model.py`** - Advanced testing with multiple options
- **`README_TESTING.md`** - This file

## 🎯 What Each Script Does

### demo.py
- Shows how the model works
- Tests with a sample image
- Provides code examples for integration

### simple_test.py
- Quick testing of single images or sample dataset
- Simple output format
- Good for basic validation

### test_model.py
- Comprehensive testing options
- Performance metrics
- Batch processing
- Interactive mode
- Detailed statistics

## 📊 Understanding the Results

The model outputs:
- **"Person detected"** - The image contains a person
- **"No person detected"** - The image does not contain a person
- **Confidence score** - How confident the model is (0.0 to 1.0)
- **Prediction time** - How long the prediction took

## 🔧 Requirements

Make sure you have the required packages:
```bash
pip install tensorflow pillow numpy matplotlib
```

## 🎮 Example Usage

```bash
# Quick demo
python demo.py

# Test your own image
python simple_test.py my_photo.jpg

# Test all images in a folder
python test_model.py --directory /path/to/my/images/

# Get detailed performance metrics
python test_model.py --batch-test
```

## 🐛 Troubleshooting

1. **Model not found**: Make sure `people_detection_model.h5` exists in the directory
2. **Import errors**: Install required packages with pip
3. **Image errors**: Make sure image files are valid and accessible
4. **Memory issues**: For large batches, test smaller groups of images

## 📈 Performance Tips

- The model works best with images that contain clear, visible people
- Images are automatically resized to 224x224 pixels
- Average prediction time is typically 0.1-0.5 seconds per image
- For batch processing, the model can handle hundreds of images efficiently
