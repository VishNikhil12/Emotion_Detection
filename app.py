import tensorflow as tf
from tensorflow.keras.layers import InputLayer as OriginalInputLayer

# --- Safe Monkey-Patch for InputLayer (if still needed) ---
_original_input_layer_init = OriginalInputLayer.__init__
_original_from_config = OriginalInputLayer.from_config

def patched_input_layer_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    _original_input_layer_init(self, *args, **kwargs)

@classmethod
def patched_from_config(cls, config):
    if 'batch_shape' in config:
        config['batch_input_shape'] = config.pop('batch_shape')
    return _original_from_config(config)

OriginalInputLayer.__init__ = patched_input_layer_init
OriginalInputLayer.from_config = patched_from_config
# --- End of Monkey-Patch ---

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😀",
    layout="centered"
)

st.title("Emotion Detection App")
st.markdown("Upload an image or use your webcam to detect emotions!")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@st.cache_resource
def load_model():
    """Load the pre-trained model and face classifier."""
    try:
        model = tf.keras.models.load_model('Custom_CNN_model.keras', compile=False)
        st.success("Custom CNN model loaded successfully!")
    except Exception as e:
        st.warning(f"Could not load the original model: {str(e)}. Using a placeholder model instead.")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
    
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, face_classifier

def preprocess_face(face):
    """Preprocess face image for model input."""
    face_resized = cv2.resize(face, (48, 48))
    if len(face_resized.shape) > 2:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_resized = face_resized.astype("float") / 255.0
    face_resized = np.expand_dims(face_resized, axis=-1)
    face_resized = np.expand_dims(face_resized, axis=0)
    return face_resized

def predict_emotion(face):
    """Predict emotion from face image."""
    try:
        processed_face = preprocess_face(face)
        prediction = model.predict(processed_face)[0]
        return prediction
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Load model and face classifier
model, face_classifier = load_model()

# Initialize session state for streaming if not already present.
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# Provide only "Upload Image" and "Real Time Streaming" options.
option = st.radio("Choose input method:", ["Upload Image", "Real Time Streaming"], key="input_option")

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload_image")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_array = np.array(image)
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        with st.spinner("Detecting emotions..."):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            result_image = frame.copy()
            if len(faces) > 0:
                st.success(f"Found {len(faces)} face(s)!")
                for (x, y, w, h) in faces:
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    prediction = predict_emotion(face)
                    if prediction is not None:
                        emotion = emotion_labels[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100
                        label = f"{emotion}: {confidence:.2f}%"
                        cv2.putText(result_image, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_image, caption="Detected Emotions", use_container_width=True)
            else:
                st.warning("No faces detected in the image.")

elif option == "Real Time Streaming":
    st.write("Real Time Streaming from your webcam")
    # Create Start and Stop buttons only once with unique keys.
    if st.button("Start Streaming", key="start_streaming"):
        st.session_state.streaming = True

    def stop_streaming_callback():
        st.session_state.streaming = False

    st.button("Stop Streaming", key="stop_streaming", on_click=stop_streaming_callback)

    streaming_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        while st.session_state.streaming:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = gray[y:y + h, x:x + w]
                prediction = predict_emotion(face)
                if prediction is not None:
                    emotion = emotion_labels[np.argmax(prediction)]
                    confidence = np.max(prediction) * 100
                    label = f"{emotion}: {confidence:.2f}%"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            streaming_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            time.sleep(0.05)
        cap.release()
        st.success("Streaming stopped.")

st.markdown("---")
st.markdown("### About this app")
st.markdown("This app uses a Custom CNN model trained to detect emotions from facial expressions.")
st.markdown("The model can detect 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.")
st.markdown("**Note:** The model works best with clear, well-lit facial images.")
