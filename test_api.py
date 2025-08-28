# aiProject/test_api.py

import requests
import os
import json

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict/" # Make sure this matches your FastAPI endpoint

# Path to a sample image you want to test
# ADJUST THIS PATH to point to an actual image in your downloaded dataset
# Example:
# IMAGE_TO_TEST = os.path.join("data", "archive", "nhcd", "nhcd", "numerals", "1", "100.png")
# Make sure this specific image file exists after you extract your dataset!
# Pick one of the images you know your model should recognize, e.g., a numeral '0' or '1'
# For example, if your dataset has a folder 'numerals/0' with 'img001-001.png' inside:
IMAGE_TO_TEST = os.path.join("data", "archive", "nhcd", "nhcd", "numerals", "0", "001_01.jpg")

# --- Send the Image for Prediction ---
def get_prediction(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}. Please check the path and ensure the dataset is correctly placed.")
        return

    print(f"Sending image: {image_path}")
    try:
        # Open the image file in binary read mode
        with open(image_path, "rb") as image_file:
            # Prepare the files dictionary for the POST request
            # 'file' here should match the parameter name in your FastAPI endpoint (e.g., `image: UploadFile`)
            files = {"file": (os.path.basename(image_path), image_file, "image/png")}
            response = requests.post(API_URL, files=files)

        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        prediction = response.json()
        print(f"Prediction successful!")
        print(f"  Filename: {prediction.get('filename')}")
        print(f"  Predicted Character: {prediction.get('predicted_character')}")
        print(f"  Confidence: {prediction.get('confidence'):.4f}")
        print(f"  Predicted Class ID: {prediction.get('predicted_class_id')}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        print(f"Check if FastAPI server is running at {API_URL}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server response: {e.response.text}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Raw response text: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    get_prediction(IMAGE_TO_TEST)