import cv2
import numpy as np
import os

print("Testing application components...")

# Check if the face classifier file exists
face_classifier_path = 'haarcascade_frontalface_default.xml'
print(f"Face classifier exists: {os.path.exists(face_classifier_path)}")

# Try to load the face classifier
try:
    face_classifier = cv2.CascadeClassifier(face_classifier_path)
    print("Face classifier loaded successfully")
except Exception as e:
    print(f"Error loading face classifier: {str(e)}")

# Check if the model file exists
model_path = 'ResNet50_Model.keras'
print(f"Model file exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# List all files in the current directory
print("\nFiles in current directory:")
for file in os.listdir('.'):
    file_size = os.path.getsize(file) / 1024  # Size in KB
    if file_size > 1024:
        print(f"- {file} ({file_size/1024:.2f} MB)")
    else:
        print(f"- {file} ({file_size:.2f} KB)")

print("\nApp is ready to be deployed to Hugging Face Spaces!")