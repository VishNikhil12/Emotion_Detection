# Hugging Face Spaces Deployment Guide

This guide will walk you through the process of deploying your Emotion Detection app to Hugging Face Spaces.

## Prerequisites

1. A Hugging Face account (sign up at https://huggingface.co/join if you don't have one)
2. Git and Git LFS installed on your computer
3. Your Emotion Detection app files ready for deployment

## Step 1: Create a New Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click on "Create new Space"
3. Fill in the following details:
   - **Owner**: Select your username or organization
   - **Space name**: Choose a name like "emotion-detection-app"
   - **SDK**: Select "Streamlit"
   - **Space hardware**: Choose "CPU" (sufficient for inference)
   - **Visibility**: Public or Private as preferred
4. Click "Create Space"

## Step 2: Clone the Space Repository

Once your Space is created, you'll need to clone the repository to your local machine:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-app
cd emotion-detection-app
```

Replace `YOUR_USERNAME` with your Hugging Face username.

## Step 3: Copy Your App Files

Copy all the necessary files from your project to the cloned repository:

```bash
# Example commands (adjust paths as needed)
cp /path/to/your/app.py .
cp /path/to/your/requirements.txt .
cp /path/to/your/README.md .
cp /path/to/your/haarcascade_frontalface_default.xml .
cp /path/to/your/Custom_CNN_model.keras .
cp /path/to/your/.gitattributes .
```

## Step 4: Set Up Git LFS for Large Files

Since your model file is large, you need to use Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.xml"
git add .gitattributes
```

## Step 5: Commit and Push Your Files

```bash
git add .
git commit -m "Initial app deployment"
git push
```

## Step 6: Monitor the Deployment

1. Go to your Space on Hugging Face (https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-app)
2. Click on the "Settings" tab
3. Go to the "Build logs" section to monitor the deployment progress

## Troubleshooting

If you encounter issues with the deployment:

1. **Model loading errors**: Make sure your model file is properly tracked with Git LFS
2. **Dependencies issues**: Check your requirements.txt file for any missing or incompatible packages
3. **Space limitations**: If your model is too large, consider using a smaller model or optimizing your current one

## PowerShell Commands (Windows)

If you're using PowerShell on Windows, use these commands instead:

```powershell
# Clone the repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/emotion-detection-app
cd emotion-detection-app

# Copy files (adjust paths as needed)
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\app.py" -Destination .
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\requirements.txt" -Destination .
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\README.md" -Destination .
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\haarcascade_frontalface_default.xml" -Destination .
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\Custom_CNN_model.keras" -Destination .
Copy-Item -Path "C:\Users\hp\OneDrive\Desktop\Emotion_Detection\.gitattributes" -Destination .

# Set up Git LFS
git lfs install
git lfs track "*.keras"
git lfs track "*.xml"
git add .gitattributes

# Commit and push
git add .
git commit -m "Initial app deployment"
git push
``` 