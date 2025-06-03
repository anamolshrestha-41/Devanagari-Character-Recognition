# api/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import json
import os

# Import your data loader functions - make sure utils/__init__.py exists
from utils.data_loader import preprocess_image

app = FastAPI(
    title="Devanagari Character Classifier API",
    description="API for classifying Devanagari characters from images.",
    version="1.0.0",
)

# --- Configuration (MUST match your model training/saving) ---
MODEL_PATH = "models/devanagari_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
MODEL_TARGET_SIZE = (64, 64) # e.g., 64x64 pixels
MODEL_GRAYSCALE = True       # Whether the model expects grayscale images
MODEL_NORMALIZE = True       # Whether pixel values should be normalized to [0, 1]

# Global variables to hold the loaded model and character map
model = None
character_map = None

@app.on_event("startup")
async def load_resources():
    """Load the model and class names when the API starts up."""
    global model, character_map

    # --- Load TensorFlow Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {os.path.abspath(MODEL_PATH)}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}!")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        # Re-raise the exception to prevent the server from starting with a broken model
        raise RuntimeError(f"Could not load model: {e}")

    # --- Load Class Names (character_map) ---
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"Error: Class names file not found at {os.path.abspath(CLASS_NAMES_PATH)}")
        raise RuntimeError(f"Class names file not found: {CLASS_NAMES_PATH}")
    try:
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            character_map = json.load(f)
        print(f"Class names loaded successfully! Found {len(character_map)} classes.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {CLASS_NAMES_PATH}: {e}")
        raise RuntimeError(f"Could not load class names (JSON error): {e}")
    except Exception as e:
        print(f"Error loading class names from {CLASS_NAMES_PATH}: {e}")
        raise RuntimeError(f"Could not load class names: {e}")

class PredictionResponse(BaseModel):
    """Pydantic model for the API response."""
    predicted_character: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_character(file: UploadFile = File(...)):
    """
    Predicts the Devanagari character from an uploaded image.
    """
    if model is None or character_map is None:
        # This check should ideally not be hit if startup event works, but good for safety
        raise HTTPException(status_code=500, detail="Model or character map not loaded. API is not ready.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image. Please upload an image file.")

    try:
        # Read image file content
        contents = await file.read()
        # Open image using Pillow and convert to RGB (standard format)
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess the image using the shared utility function
        processed_image = preprocess_image(
            image,
            target_size=MODEL_TARGET_SIZE,
            grayscale=MODEL_GRAYALE,
            normalize=MODEL_NORMALIZE
        )

        # Add a batch dimension (model expects a batch of images)
        # e.g., from (H, W, C) to (1, H, W, C)
        prediction = model.predict(np.expand_dims(processed_image, axis=0))[0]

        # Get the predicted class ID and its confidence score
        predicted_class_id = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Map the numerical class ID back to the actual Devanagari character
        if 0 <= predicted_class_id < len(character_map):
            predicted_char = character_map[predicted_class_id]
        else:
            predicted_char = "Unknown Character (ID out of bounds)" # Fallback for unexpected IDs

        # Return the prediction
        return PredictionResponse(
            predicted_character=predicted_char,
            confidence=confidence
        )

    except Exception as e:
        # Log the detailed error for debugging purposes (server-side)
        print(f"Prediction error: {e}")
        # Return a generic 500 error to the client to avoid exposing internal details
        raise HTTPException(status_code=500, detail=f"Error processing image or making prediction. Please try again or contact support if the issue persists.")

@app.get("/")
async def root():
    """Root endpoint for basic API health check."""
    return {"message": "Devanagari Character Classifier API is running!"}