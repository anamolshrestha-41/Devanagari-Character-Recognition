from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import os
import sys

# Add the project root to sys.path so utils can be imported
# This assumes app.py is in aiProject/api/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import from utils
from utils.data_loader import preprocess_image # Ensure this function exists in data_loader.py

app = FastAPI()

# Define model and class names paths relative to the project root
MODEL_PATH = os.path.join(project_root, 'models', 'devanagari_model.h5')
CLASS_NAMES_PATH = os.path.join(project_root, 'models', 'class_names.json')

model = None
class_names = []

# Model preprocessing settings - MUST match what was used during training in the notebook
IMAGE_TARGET_SIZE = (64, 64)
IMAGE_GRAYSCALE = True
IMAGE_NORMALIZE = True

# Event handler to load the model and class names when the API starts
@app.on_event("startup")
async def load_resources():
    global model, class_names
    print("Attempting to load model and class names...")
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        # Raise a runtime error to prevent the app from starting if model fails to load
        print(f"Error loading model from {MODEL_PATH}: {e}")
        raise RuntimeError(f"Could not load model: {e}")

    try:
        # Load class names
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"Class names loaded successfully from {CLASS_NAMES_PATH}")
    except Exception as e:
        # Raise a runtime error if class names fail to load
        print(f"Error loading class names from {CLASS_NAMES_PATH}: {e}")
        raise RuntimeError(f"Could not load class names: {e}")

    print(f"Model input shape: {model.input_shape}")
    print(f"Number of classes in loaded model: {model.output_shape[-1]}")
    if len(class_names) > 0:
        print(f"First few loaded class names: {class_names[:5]}")
    else:
        print("No class names loaded.")


@app.get("/")
async def root():
    return {"message": "Devanagari Character Classification API. Visit /docs for more info."}

# Define the prediction endpoint
@app.post("/predict/")
async def predict_character(file: UploadFile = File(...)):
    if model is None:
        # This should ideally not happen if startup event handled correctly, but good for safety
        raise HTTPException(status_code=500, detail="Model not loaded. Server startup error.")
    if not class_names:
        raise HTTPException(status_code=500, detail="Class names not loaded. Server startup error.")

    try:
        # Read the image bytes from the uploaded file
        image_bytes = await file.read()
        # Open the image using PIL (Pillow)
        image_pil = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image using the same utility function as the notebook
        processed_image = preprocess_image(
            image_pil, # Pass the PIL Image object directly
            target_size=IMAGE_TARGET_SIZE,
            grayscale=IMAGE_GRAYSCALE,
            normalize=IMAGE_NORMALIZE
        )

        # Expand dimensions to create a batch of 1 image
        # Model expects input in batch format (batch_size, height, width, channels)
        input_image = np.expand_dims(processed_image, axis=0)

        # Make prediction
        predictions = model.predict(input_image)
        predicted_class_id = np.argmax(predictions[0])

        if predicted_class_id >= len(class_names):
            raise ValueError(f"Predicted class ID {predicted_class_id} is out of bounds for class_names (size {len(class_names)}).")

        predicted_character = class_names[predicted_class_id]
        confidence = float(np.max(predictions[0]))

        return {
            "filename": file.filename,
            "predicted_character": predicted_character,
            "confidence": confidence,
            "predicted_class_id": int(predicted_class_id) # Convert to standard int for JSON
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process image or make prediction: {e}")