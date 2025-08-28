import os
import numpy as np
from PIL import Image
import cv2 # For loading images if not using PIL for all cases
import tensorflow as tf # For preprocessing, e.g., tf.image functions

# Define a mapping from folder names (as they appear in your dataset)
# to Devanagari characters. You MUST verify these against your actual dataset's folder names.
# Add any missing mappings based on the warnings you saw.
# Numerals mapping
NUMERAL_MAPPING = {
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'
}

# Consonants mapping
CONSONANT_MAPPING = {
    '1': 'क', '2': 'ख', '3': 'ग', '4': 'घ', '5': 'ङ',
    '6': 'च', '7': 'छ', '8': 'ज', '9': 'झ', '10': 'ञ',
    '11': 'ट', '12': 'ठ', '13': 'ड', '14': 'ढ', '15': 'ण',
    '16': 'त', '17': 'थ', '18': 'द', '19': 'ध', '20': 'न',
    '21': 'प', '22': 'फ', '23': 'ब', '24': 'भ', '25': 'म',
    '26': 'य', '27': 'र', '28': 'ल', '29': 'व', '30': 'श',
    '31': 'ष', '32': 'स', '33': 'ह', '34': 'क्ष', '35': 'त्र', '36': 'ज्ञ'
}

# Vowels mapping
VOWEL_MAPPING = {
    '1': 'अ', '2': 'आ', '3': 'इ', '4': 'ई', '5': 'उ',
    '6': 'ऊ', '7': 'ऋ', '8': 'ए', '9': 'ऐ', '10': 'ओ',
    '11': 'औ', '12': 'अं', '13': 'अः'
}

FOLDER_NAME_TO_DEVANAGARI_CHAR = {}


def preprocess_image(image_input, target_size=(64, 64), grayscale=True, normalize=True):
    """
    Preprocesses a single image for model input.

    Args:
        image_input: Can be a file path (str), a NumPy array (OpenCV format), or a PIL Image object.
        target_size (tuple): Desired (height, width) for the image.
        grayscale (bool): Whether to convert the image to grayscale.
        normalize (bool): Whether to normalize pixel values to [0, 1].

    Returns:
        np.array: Preprocessed image as a NumPy array, ready for model input.
    """
    if isinstance(image_input, str):
        # If input is a file path, load using PIL or OpenCV
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        try:
            # Use PIL for loading (more robust for various image types)
            image = Image.open(image_input).convert('RGB')
        except Exception as e:
            raise IOError(f"Could not load image from {image_input}: {e}")
    elif isinstance(image_input, np.ndarray):
        # If input is a NumPy array (e.g., from OpenCV), convert to PIL Image for consistent processing
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    elif isinstance(image_input, Image.Image):
        # If input is already a PIL Image
        image = image_input.convert('RGB')
    else:
        raise TypeError("image_input must be a file path (str), NumPy array, or PIL Image object.")

    # Resize the image
    image = image.resize(target_size, Image.LANCZOS) # Use LANCZOS for high-quality downsampling

    # Convert to grayscale if required
    if grayscale:
        image = image.convert('L') # Convert to single channel grayscale

    # Convert to NumPy array
    # If grayscale, shape will be (height, width)
    # If not grayscale (RGB), shape will be (height, width, 3)
    image_array = np.array(image)

    # Add channel dimension if grayscale (from (H, W) to (H, W, 1))
    if grayscale:
        image_array = np.expand_dims(image_array, axis=-1) # Add channel dimension

    # Normalize pixel values
    if normalize:
        image_array = image_array / 255.0

    return image_array

# Update the mapping based on category
def get_character_mapping(category_name, folder_name):
    if category_name == 'numerals':
        return NUMERAL_MAPPING.get(folder_name)
    elif category_name == 'consonants':
        return CONSONANT_MAPPING.get(folder_name)
    elif category_name == 'vowels':
        return VOWEL_MAPPING.get(folder_name)
    return None

def load_devanagari_dataset(data_root_dir_absolute, target_size=(64, 64), grayscale=True, normalize=True):
    """
    Loads and preprocesses the Devanagari character dataset from the specified root directory.

    Args:
        data_root_dir_absolute (str): The absolute path to the 'nhcd/nhcd' directory
                                      containing 'consonants', 'numerals', 'vowels' subfolders.
        target_size (tuple): Desired (height, width) for image preprocessing.
        grayscale (bool): Whether to convert images to grayscale.
        normalize (bool): Whether to normalize pixel values to [0, 1].

    Returns:
        tuple: (images (np.array), labels (np.array), class_names (list))
               images: Stacked preprocessed image arrays.
               labels: Integer labels corresponding to each image.
               class_names: List of Devanagari characters in the order of their assigned labels.
    """
    all_images = []
    all_labels = []
    class_name_to_label = {} # Maps Devanagari char to integer label
    label_to_class_name = [] # Maps integer label to Devanagari char

    current_label = 0

    if not os.path.exists(data_root_dir_absolute):
        raise FileNotFoundError(f"Data root directory not found: {data_root_dir_absolute}")
    if not os.path.isdir(data_root_dir_absolute):
        raise NotADirectoryError(f"Data root path is not a directory: {data_root_dir_absolute}")

    # Categories are usually 'consonants', 'numerals', 'vowels'
    # Iterate through these primary categories
    for category_name in os.listdir(data_root_dir_absolute):
        category_path = os.path.join(data_root_dir_absolute, category_name)

        if not os.path.isdir(category_path):
            continue # Skip any non-directory files like .DS_Store or READMEs

        print(f"Processing category: {category_path}")

        # Iterate through subfolders (e.g., '0', '1', 'क', 'ख') within each category
        for folder_name in sorted(os.listdir(category_path)):
            folder_path = os.path.join(category_path, folder_name)

            if not os.path.isdir(folder_path):
                continue

            # Map folder name to Devanagari character based on category
            devanagari_char = get_character_mapping(category_name, folder_name)

            if devanagari_char is None:
                print(f"    WARNING: Folder name '{folder_name}' in category '{category_name}' not found in mapping. Skipping this folder.")
                continue

            # Assign a unique integer label if this character is new
            if devanagari_char not in class_name_to_label:
                class_name_to_label[devanagari_char] = current_label
                label_to_class_name.append(devanagari_char)
                current_label += 1

            label = class_name_to_label[devanagari_char]
            images_in_folder = 0

            # Iterate through image files in the current character folder
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        # Use the universal preprocess_image function
                        processed_img = preprocess_image(
                            image_path,
                            target_size=target_size,
                            grayscale=grayscale,
                            normalize=normalize
                        )
                        all_images.append(processed_img)
                        all_labels.append(label)
                        images_in_folder += 1
                    except (IOError, FileNotFoundError, TypeError) as e:
                        print(f"        ERROR: Could not process image {image_path}: {e}")
                        continue
            if images_in_folder > 0:
                print(f"    Loaded {images_in_folder} images from folder '{folder_name}' (Devanagari: '{devanagari_char}').")
            else:
                print(f"    No images loaded from folder '{folder_name}' (Devanagari: '{devanagari_char}').")


    if not all_images:
        raise ValueError("No data loaded. Please check data_root_dir_absolute and FOLDER_NAME_TO_DEVANAGARI_CHAR mapping.")

    # Convert lists to numpy arrays
    images_np = np.array(all_images)
    labels_np = np.array(all_labels)

    # Ensure class_names list is sorted by label ID
    # This is implicitly handled by `label_to_class_name.append(devanagari_char)`
    # if `current_label` increments correctly.
    # But if you want to be extra safe, sort based on actual labels.
    # For simplicity, if `label_to_class_name` builds correctly, it should be ordered.
    final_class_names = [item[0] for item in sorted(class_name_to_label.items(), key=lambda x: x[1])]


    print(f"\nSuccessfully loaded {len(all_images)} images across {len(final_class_names)} unique classes.")
    print(f"Unique characters found: {', '.join(final_class_names)}")
    print(f"Image shape after preprocessing: {images_np.shape[1:]}")

    return images_np, labels_np, final_class_names