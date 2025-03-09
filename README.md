# Emotion Detection App

This application uses a custom CNN deep learning model to detect emotions from facial expressions in images and in real time.

```
title: Emotion Detection App
emoji: 😀
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.22.0
app_file: app.py
pinned: false
license: mit
```

## Features

- Upload an image **or** stream video in real time from your webcam
- Detect faces in the input
- Analyze emotions for each detected face
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

1. Choose whether to upload an image or use real-time webcam streaming.
2. If uploading, select an image file from your device.
3. If using real-time streaming, allow camera access in your browser.
4. The app will process the input and display the results.
5. Detected faces are highlighted with a green bounding box and labeled with the predicted emotion and confidence.

## Technical Details

- **Built with:** Streamlit  
- **Face Detection:** OpenCV with Haar Cascade Classifier  
- **Emotion Classification:** A custom CNN model trained on facial expression datasets  
- **Input:** The model processes 48x48 grayscale face images  
- **Deployment:** Hosted on Hugging Face Spaces using WebRTC for real-time webcam streaming

## Model Information

The custom CNN model was trained on facial expression datasets to classify emotions from grayscale face images. The model accepts 48x48 grayscale images as input and outputs probabilities for each of the 7 emotion classes.

## Credits

Created by Vishwanath Nikhil
