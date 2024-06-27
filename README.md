# Plant Disease Detector

PlantDiseaseDetector is a machine learning project designed to assist farmers in detecting crop diseases using computer vision. The project leverages YOLOv8 for real-time object detection and segmentation, enabling efficient monitoring and management of agricultural fields. Additionally, a Keras-based neural network is used for detailed classification of detected leaves to identify specific diseases.

## Features

- **Crop Disease Detection**: Identify various diseases in crops using images of leaves.
- **Real-Time Segmentation**: Segment and highlight diseased areas on leaves in real-time using YOLOv5.
- **Multi-Leaf Detection**: Handle multiple leaves in a single image, ensuring comprehensive analysis.
- **Keras-based Classification**: Utilize a Keras model to classify segmented leaves into specific disease categories.
- **User-Friendly Web Interface**: A Flask-based web application with endpoints for video streaming and image uploads.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Endpoints](#endpoints)
4. [Results](#results)
5. [License](#license)

## Installation

To get started with PlantDiseaseDetector, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/PlantDiseaseDetector.git
    cd PlantDiseaseDetector
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv leafclassification
    leafclassification\Scripts\activate  # On Windows
    source leafclassification/bin/activate  # On macOS/Linux
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Have the YOLOv8 weights in the project directory**:
    - Ensure you have `best.pt` in the root directory.

5. **Have the Keras model in the project directory**:
    - Ensure you have `leaf_classification_model.keras` in the root directory.

## Usage

To run the Flask web application:

1. **Navigate to the Flask application directory**:
    ```bash
    cd flask-yolo-app
    ```

2. **Start the Flask server**:
    ```bash
    python app.py
    ```

3. **Open your browser and navigate to**:
    ```
    http://127.0.0.1:5000/
    ```

## Endpoints

The web application provides the following endpoints:

- **Home**: `/`
  - The landing page with options to navigate to different functionalities.
  
- **Video Stream**: `/video`
  - Stream real-time video and perform disease detection on-the-fly.
  
- **Upload Image**: `/upload`
  - Upload an image of leaves for disease detection and segmentation.

## Results

- **Segmentation and Detection**:
  - YOLOv8 effectively segments and detects leaves in real-time.
- **Classification**:
  - The Keras model classifies leaves into specific disease categories with high accuracy.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
