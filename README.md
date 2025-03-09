# Emotion Detection App

This application uses a Custom CNN deep learning model to detect emotions from facial expressions in images.

---
title: Emotion Detection App
emoji: 😀
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.22.0
app_file: app.py
pinned: false
license: mit
---

## Features

- Upload an image or take a photo with your webcam
- Detect faces in the image
- Analyze emotions in each detected face
- Display results with bounding boxes and emotion labels

## Emotions Detected

The model can detect 7 different emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## How to Use

1. Choose whether to upload an image or take a photo
2. If uploading, select an image file from your device
3. If using webcam, take a photo when ready
4. The app will process the image and display the results
5. Each detected face will be highlighted with a green box and labeled with the detected emotion

## Technical Details

- Built with Streamlit
- Uses OpenCV for face detection with Haar Cascade Classifier
- Emotion classification with a ResNet50 model trained on facial expression datasets
- Deployed on Hugging Face Spaces

## Model Information

The Custom CNN model was trained on facial expression datasets to classify emotions from grayscale face images. The model takes a 224x224 grayscale image as input and outputs probabilities for each of the 7 emotion classes.

## Credits

Created by Vishwanath Nikhil 