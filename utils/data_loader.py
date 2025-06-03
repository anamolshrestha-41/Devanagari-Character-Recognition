# utils/data_loader.py
import cv2 # Often used for image operations, especially if you handle non-standard formats
from PIL import Image
import numpy as np
import os
import tensorflow as tf # Used for tf.keras.utils.to_categorical
from sklearn.model_selection import train_test_split # Used for splitting dataset

def preprocess_image(image_input, target_size=(64, 64), grayscale=True, normalize=True):
    """
    Preprocesses an image to be ready for model input.
    This function should be identical to the one used during model training.

    Args:
        image_input: Can be a PIL Image object or a NumPy array.
        target_size (tuple): Desired (width, height) of the output image.
        grayscale (bool): Whether to convert the image to grayscale.
        normalize (bool): Whether to normalize pixel values from [0, 255] to [0, 1].

    Returns:
        numpy.ndarray: The preprocessed image array with shape (height, width, channels),
                       ready for model input (after adding a batch dimension).
    """
    # Ensure input is a PIL Image object for consistent processing
    if isinstance(image_input, np.ndarray):
        # Assuming BGR for OpenCV arrays, convert to RGB for PIL
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif image_input.ndim == 2: # Grayscale NumPy array
            image = Image.fromarray(image_input)
        else:
            raise TypeError("Unsupported NumPy array format for image_input.")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise TypeError("image_input must be a PIL Image or NumPy array.")

    # Resize the image
    image = image.resize(target_size, Image.LANCZOS) # Use LANCZOS for high-quality downsampling

    # Convert to grayscale if required
    if grayscale:
        image = image.convert('L') # 'L' mode for grayscale

    # Convert PIL Image to NumPy array
    img_array = np.array(image)

    # Add channel dimension if it's missing (e.g., for grayscale images)
    # Model expects (height, width, channels)
    if img_array.ndim == 2: # Grayscale image
        img_array = np.expand_dims(img_array, axis=-1) # Becomes (height, width, 1)
    elif img_array.ndim == 3 and img_array.shape[-1] == 4: # RGBA image
        img_array = img_array[..., :3] # Remove alpha channel if not grayscale

    # Normalize pixel values to [0, 1]
    if normalize:
        img_array = img_array / 255.0

    return img_array

# --- IMPORTANT: Define your 58 Devanagari characters in the EXACT order of your labels ---
# This list MUST match the order used to assign integer IDs during model training.
# You need to fill this out completely based on your dataset's classes.
# The order here determines the class_id (index).
ALL_DEVANAGARI_CHARS = [
    # 36 Consonants (Example, YOU NEED TO LIST ALL 36 IN YOUR DATASET'S ORDER)
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श',
    'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', # Example for compound consonants, verify if your dataset has these

    # 12 Vowels (Example, YOU NEED TO LIST ALL 12 IN YOUR DATASET'S ORDER)
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'अं', # Common 12 vowels including Anusvara for some datasets

    # 10 Numbers (All 10 - Standard Unicode)
    '०', '१', '२', '३', '४', '५', '६', '७', '८', '९'
]

# This dictionary maps the textual folder names (or inferred character names from filenames)
# to their corresponding Devanagari Unicode characters.
# You MUST verify and complete this mapping based on your specific dataset's folder names/naming conventions.
# Use .lower() on folder names to handle case inconsistencies.
FOLDER_NAME_TO_DEVANAGARI_CHAR = {
    # Consonants (Example mappings - VERIFY WITH YOUR ACTUAL FOLDER NAMES)
    'ka': 'क', 'kha': 'ख', 'ga': 'ग', 'gha': 'घ', 'nga': 'ङ',
    'cha': 'च', 'chha': 'छ', 'ja': 'ज', 'jha': 'झ', 'nya': 'ञ',
    'tta': 'ट', 'ttha': 'ठ', 'dda': 'ड', 'ddha': 'ढ', 'nna': 'ण', # These names might vary significantly in your dataset, e.g., 'ta_retroflex'
    'ta': 'त', 'tha': 'थ', 'da': 'द', 'dha': 'ध', 'na': 'न',
    'pa': 'प', 'pha': 'फ', 'ba': 'ब', 'bha': 'भ', 'ma': 'म',
    'ya': 'य', 'ra': 'र', 'la': 'ल', 'va': 'व', 'sha': 'श',
    'shha': 'ष', 'sa': 'स', 'ha': 'ह',
    'ksha': 'क्ष', 'tra': 'त्र', 'gya': 'ज्ञ', # Common compound characters - include only if your dataset has them

    # Vowels (Example mappings - VERIFY WITH YOUR ACTUAL FOLDER NAMES)
    'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ee': 'ई', 'u': 'उ', 'oo': 'ऊ',
    'rri': 'ऋ', 'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ', 'am': 'अं', # Folder names might be like 'a_vowel', 'aa_vowel', 'anusvara' etc.

    # Numbers (Example mappings - VERIFY WITH YOUR ACTUAL FOLDER NAMES)
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९'
}


def load_devanagari_dataset(data_root_dir, target_size=(64, 64), grayscale=True, normalize=True):
    """
    Loads the Devanagari character dataset from the specified directory structure,
    preprocesses images, and assigns one-hot encoded labels.

    Args:
        data_root_dir (str): Path to the root directory containing 'consonants', 'numerals', 'vowels' folders.
                             E.g., 'data/archive/nhcd/nhcd'
        target_size (tuple): Desired (width, height) for image preprocessing.
        grayscale (bool): Whether to convert images to grayscale.
        normalize (bool): Whether to normalize pixel values to [0, 1].

    Returns:
        tuple: (images, labels_one_hot, class_names, class_to_id_map)
            images (np.ndarray): Array of preprocessed images.
            labels_one_hot (np.ndarray): One-hot encoded labels.
            class_names (list): List of character names corresponding to class IDs (same as ALL_DEVANAGARI_CHARS).
            class_to_id_map (dict): Dictionary mapping character names to class IDs.
    """
    images = []
    labels = [] # Integer labels

    # Build the class_to_id_map based on ALL_DEVANAGARI_CHARS order
    class_to_id_map = {char: idx for idx, char in enumerate(ALL_DEVANAGARI_CHARS)}
    class_names = ALL_DEVANAGARI_CHARS # This will be the list passed to the API

    # Assuming the data structure is data_root_dir/[category]/[character_folder]/image.png
    categories = ['consonants', 'numerals', 'vowels']
    total_images_loaded = 0

    print(f"Starting dataset loading from: {os.path.abspath(data_root_dir)}")

    for category in categories:
        category_path = os.path.join(data_root_dir, category)
        if not os.path.isdir(category_path):
            print(f"  WARNING: Category directory '{category_path}' not found. Skipping '{category}'.")
            continue
        else:
            print(f"  Processing category: {category_path}")

        # Iterate through character subfolders (e.g., 'ka', '0', 'a', 'aa')
        for char_folder_name in os.listdir(category_path):
            char_folder_path = os.path.join(category_path, char_folder_name)
            if not os.path.isdir(char_folder_path): # Skip non-directories
                continue

            # Get the actual Devanagari character from the folder name using your mapping
            devanagari_char = FOLDER_NAME_TO_DEVANAGARI_CHAR.get(char_folder_name.lower()) # Use .lower() for robust matching
            if devanagari_char is None:
                print(f"    WARNING: Folder name '{char_folder_name}' not found in FOLDER_NAME_TO_DEVANAGARI_CHAR mapping. Skipping this folder.")
                continue

            # Get the integer class ID for this Devanagari character
            class_id = class_to_id_map.get(devanagari_char)
            if class_id is None:
                print(f"    WARNING: Devanagari character '{devanagari_char}' (from folder '{char_folder_name}') not found in ALL_DEVANAGARI_CHARS list. Skipping this folder.")
                continue

            # Load images from the character folder
            images_in_folder = 0
            for image_filename in os.listdir(char_folder_path):
                # Ensure it's an image file by checking common extensions
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(char_folder_path, image_filename)
                    try:
                        # Load and preprocess the image
                        img = Image.open(image_path).convert('RGB') # Load as RGB first
                        preprocessed_img = preprocess_image(img, target_size, grayscale, normalize)
                        images.append(preprocessed_img)
                        labels.append(class_id)
                        images_in_folder += 1
                        total_images_loaded += 1
                    except Exception as e:
                        print(f"      ERROR: Could not load or preprocess image {image_path}: {e}")
            print(f"    Loaded {images_in_folder} images from folder '{char_folder_name}' (Devanagari: '{devanagari_char}').")

    # Convert lists to NumPy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)

    # One-hot encode the integer labels
    if len(labels_array) > 0:
        labels_one_hot = tf.keras.utils.to_categorical(labels_array, num_classes=len(ALL_DEVANAGARI_CHARS))
        print(f"Successfully loaded {total_images_loaded} images across {len(class_names)} classes.")
    else:
        labels_one_hot = np.array([]) # Return empty array if no labels were loaded
        print(f"WARNING: No images were loaded from {os.path.abspath(data_root_dir)}. Please check your DATA_ROOT_DIR, folder names, and mappings.")

    return images_array, labels_one_hot, class_names, class_to_id_map