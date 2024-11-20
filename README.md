# Real Time Sign Language Detection

This repository contains scripts and resources for building a real-time sign language detection system using machine learning. The system captures and classifies hand gestures for five different signs: **Hello**, **OK**, **I Love You**, **See You Again**, and **Thank You**.

## Repository Contents

- **`Dataset Updation.py`**: A script for adding new data to the existing dataset, allowing for incremental updates by capturing additional images for specified hand signs and appending them to the dataset.
- **`Primary Dataset.py`**: The primary script to create the initial dataset by capturing keypoints for each sign using MediaPipe and storing them in a CSV file.
- **`Sign language detection.py`**: The main script for real-time sign language detection. This script uses OpenCV and a trained model to recognize hand signs from the live camera feed.
- **`sign_language_classification_model.h5`**: The pre-trained model file for sign classification. It uses a neural network to classify the gestures based on hand keypoints.
- **`sign_language_keypoints.csv`**: The dataset file containing keypoint data for each sign, along with their labels, used for training the model.
- **`Training`**: A folder (or script) for training the neural network model on the sign language keypoints dataset.

## Key Features

- **Real-time Gesture Detection**: Uses OpenCV to capture video from the camera and detect hand signs in real-time.
- **MediaPipe Integration**: Leverages MediaPipe to extract hand keypoints, providing accurate input data for gesture recognition.
- **Incremental Dataset Updating**: Allows for continuous dataset improvements by adding new images, enhancing model accuracy over time.

## Getting Started

1. **Create the Initial Dataset**: Run `Primary Dataset.py` to capture and save the initial keypoints dataset.
2. **Train the Model**: Use the dataset to train the model by running scripts or using the files in the `Training` folder.
3. **Real-Time Detection**: Run `Sign language detection.py` to perform real-time sign language detection from the camera feed.
4. **Incremental Dataset Update** (Optional): Run `Dataset Updation.py` to add more data for specific signs.

## Requirements

- **Python 3.7+**
- **OpenCV**
- **TensorFlow**
- **MediaPipe**
- **NumPy, Pandas, Scikit-learn**

Install dependencies with:

```bash
pip install numpy pandas opencv-python tensorflow mediapipe scikit-learn
