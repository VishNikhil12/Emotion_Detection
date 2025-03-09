import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Set page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😀",
    layout="centered"
)

# App title and description
st.title("Emotion Detection App")
st.markdown("Upload an image or take a photo to detect emotions!")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_model():
    """Load the pre-trained model and face classifier"""
    try:
        # Try to load the model
        model = tf.keras.models.load_model('Custom_CNN_model.keras')
        st.success("Custom CNN model loaded successfully!")
    except Exception as e:
        st.warning("Could not load the original model. Using a placeholder model instead.")
        # Create a simple placeholder model for demonstration
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
    
    # Load face classifier
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, face_classifier

# Load model and face classifier
model, face_classifier = load_model()

# Function to predict emotion
def predict_emotion(face):
    """Predict emotion from face image"""
    # If using the original model
    if len(model.layers) > 10:  # This is a simple check to see if we're using the original model
        face = np.expand_dims(face, axis=0)
        return model.predict(face)[0]
    else:
        # Using placeholder model - return random predictions for demo
        import random
        predictions = np.zeros(7)
        max_index = random.randint(0, 6)
        predictions[max_index] = 0.7
        
        # Add some random values for other emotions
        for i in range(7):
            if i != max_index:
                predictions[i] = random.random() * 0.3
        
        # Normalize to ensure sum is 1
        predictions = predictions / np.sum(predictions)
        return predictions

# Create file uploader and camera input
option = st.radio("Choose input method:", ["Upload Image", "Take Photo"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to OpenCV format
        image_array = np.array(image)
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Detecting emotions..."):
            # Process the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            
            result_image = frame.copy()
            
            if len(faces) > 0:
                st.success(f"Found {len(faces)} face(s)!")
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (224, 224))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    
                    prediction = predict_emotion(face)
                    emotion = emotion_labels[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    
                    label = f"{emotion}: {confidence:.2f}%"
                    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert back to RGB for display
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_image, caption="Detected Emotions", use_column_width=True)
            else:
                st.warning("No faces detected in the image.")

elif option == "Take Photo":
    st.write("Take a photo with your webcam")
    picture = st.camera_input("Take a picture")
    
    if picture is not None:
        image = Image.open(picture)
        
        # Convert PIL Image to OpenCV format
        image_array = np.array(image)
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Detecting emotions..."):
            # Process the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            
            result_image = frame.copy()
            
            if len(faces) > 0:
                st.success(f"Found {len(faces)} face(s)!")
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (224, 224))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    
                    prediction = predict_emotion(face)
                    emotion = emotion_labels[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    
                    label = f"{emotion}: {confidence:.2f}%"
                    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert back to RGB for display
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_image, caption="Detected Emotions", use_column_width=True)
            else:
                st.warning("No faces detected in the image.")

# Add information about the app
st.markdown("---")
st.markdown("### About this app")
st.markdown("This app uses a Custom CNN model trained to detect emotions from facial expressions.")
st.markdown("The model can detect 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.")
st.markdown("**Note:** If the original model cannot be loaded, a placeholder model will be used for demonstration purposes.")
