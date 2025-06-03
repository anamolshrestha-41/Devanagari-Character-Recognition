# Devanagari Character Classification

## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) model to classify handwritten Devanagari characters. The system includes a Jupyter Notebook for model training and evaluation, and a FastAPI application to serve the trained model as a REST API for real-time predictions.

## Features

* **Data Loading & Preprocessing**: Efficiently loads and preprocesses Devanagari character images.
* **CNN Model**: Implements a deep Convolutional Neural Network for robust classification.
* **Training & Evaluation**: Jupyter Notebook for training the model, visualizing training history, and evaluating performance with classification reports and confusion matrices.
* **Model Persistence**: Saves the trained model and class mappings for later use.
* **FastAPI Integration**: Provides a lightweight and high-performance REST API to classify new handwritten Devanagari images.
* **Dockerization (Future)**: (Optional, could be added later) Containerize the API for easy deployment.

## Project Structure

* `aiProject/` (Your project root directory)
    * `api/`
        * `app.py` # FastAPI application to serve the model
        * `requirements.txt` # Python dependencies for the API
    * `data/`
        * `archive/`
            * `nhcd/`
                * `nhcd/` # Root directory of the Devanagari dataset (nhcd dataset)
                    * `consonants/` # Contains folders for individual consonant characters (e.g., 'ka', 'kha')
                    * `numerals/` # Contains folders for individual numeral characters (e.g., '0', '1')
                    * `vowels/` # Contains folders for individual vowel characters (e.g., 'a', 'aa')
                * `labels.csv` # (Optional: If your dataset includes a labels file)
    * `models/`
        * `devanagari_model.h5` # Trained Keras model (will be generated after notebook execution)
        * `class_names.json` # JSON file mapping class IDs to Devanagari characters (generated)
    * `notebooks/`
        * `devanagari_classification.ipynb` # Jupyter Notebook for model training and evaluation
        * `requirements.txt` # Python dependencies for the notebook
    * `utils/`
        * `__init__.py` # Makes 'utils' a Python package
        * `data_loader.py` # Utility functions for loading and preprocessing the dataset
    * `venv/` # Python Virtual Environment (ignored by Git)
    * `.gitignore` # Specifies intentionally untracked files to ignore by Git
    * `README.md` # This project's README file
 
## Dataset

This project utilizes a subset of the **Nepali Handwritten Character Dataset (NHCD)**.
The dataset is expected to be extracted into the `data/archive/nhcd/nhcd/` directory, containing `consonants`, `numerals`, and `vowels` subdirectories, which in turn contain individual character folders and image files.

**Note**: The raw dataset files are NOT committed to this repository due to their size. You will need to download and extract the dataset yourself.

**To get the dataset:**
1.  Obtain the NHCD dataset from its source (e.g., Kaggle, if applicable).
2.  Extract the contents such that the `consonants`, `numerals`, and `vowels` folders are located at: `aiProject/data/archive/nhcd/nhcd/`.

## Setup and Installation

Follow these steps to set up the project environment:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd Devanagari-Character-Classification # or whatever your aiProject folder is named
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    (Your terminal prompt should now show `(venv)` indicating the environment is active.)

4.  **Install Dependencies:**
    Install all required Python packages for both the notebook and the API.
    ```bash
    pip install -r notebooks/requirements.txt
    pip install -r api/requirements.txt
    ```

5.  **Download and Place the Dataset:**
    As mentioned in the "Dataset" section, download the Nepali Handwritten Character Dataset (NHCD) and place it in the `aiProject/data/archive/nhcd/nhcd/` directory. Ensure the structure is `aiProject/data/archive/nhcd/nhcd/{consonants, numerals, vowels}/...`.

## How to Run the Jupyter Notebook

The Jupyter Notebook (`devanagari_classification.ipynb`) handles data loading, preprocessing, model training, evaluation, and saving the trained model.

1.  **Start VS Code from the activated `venv` terminal:**
    ```bash
    code .
    ```
2.  **Open the Notebook:** In VS Code, open `notebooks/devanagari_classification.ipynb`.
3.  **Select the `venv` Kernel:** In the top right corner of the Jupyter Notebook interface, ensure that your `(venv)` Python environment is selected as the kernel (e.g., `Python 3.x.x ('venv': venv)`).
4.  **Run All Cells:** Go to `Run` -> `Run All Cells` (or click the "Run All" button in the notebook toolbar).
    * **Important:** Pay attention to the output of Cell 2, which verifies the dataset path. It should show `True` for both `Does full dataset path exist?` and `Is full dataset path a directory?`.
    * You might see warnings about folder names not found in `FOLDER_NAME_TO_DEVANAGARI_CHAR` mapping. These indicate characters in your dataset that are not currently mapped; the model will still train on the available mapped characters.
5.  **Model and Class Names Saved:** Upon successful execution, the trained model (`devanagari_model.h5`) and class names (`class_names.json`) will be saved in the `models/` directory.

## How to Run the FastAPI Application

Once the model is trained and saved by the Jupyter Notebook, you can run the FastAPI application to serve predictions.

1.  **Ensure your `venv` is active** in your terminal (same as in setup).
2.  **Navigate to the project root directory** (`aiProject`).
3.  **Run the FastAPI application:**
    ```bash
    uvicorn api.app:app --reload
    ```
4.  **Access the API:**
    * The API will typically run on `http://127.0.0.1:8000`.
    * You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.
    * You can also test the API directly using tools like Postman or `curl`. The endpoint for prediction will likely be `/predict/`.

## Dependencies

Refer to the `requirements.txt` files in the `api/` and `notebooks/` directories for a complete list of dependencies. Key dependencies include:

* `tensorflow`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `fastapi`
* `uvicorn`
* `Pillow` (PIL)
* `opencv-python`

## License

This project is licensed under the [MIT License](LICENSE). (Create a `LICENSE` file if you want to explicitly define this.)

## Contact

For any questions or suggestions, feel free to contact:
Name: Anamol Shrestha
