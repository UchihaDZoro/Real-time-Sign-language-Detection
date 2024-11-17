#Real Time Sign Language Detection
This repository contains scripts and resources for building a sign language detection system using machine learning. The system captures and classifies hand gestures for five different signs: Hello, OK, I Love You, See You Again, and Thank You.

Repository Contents
Dataset Updation.py: A script for adding new data to the existing dataset. This allows for incremental updates by capturing additional images for specified hand signs and appending them to the dataset.

Primary Dataset.py: The primary script to create the initial dataset by capturing keypoints for each sign using MediaPipe and storing them in a CSV file.

Sign language detection.py: The main script for real-time sign language detection using OpenCV and a trained model to recognize hand signs from the live camera feed.

sign_language_classification_model.h5: The pre-trained model file for sign classification. It uses a neural network to classify the gestures based on hand keypoints.

sign_language_keypoints.csv: The dataset file containing keypoint data for each sign, along with their labels, used for training the model.

Training: A folder (or script) for training the neural network model on the sign language keypoints dataset.

Key Features
Real-time Gesture Detection: Uses OpenCV to capture video from the camera and detect hand signs in real-time.
MediaPipe Integration: Leverages MediaPipe to extract hand keypoints, providing accurate input data for gesture recognition.
Incremental Dataset Updating: Allows for continuous dataset improvements with new images, enhancing model accuracy over time.
Getting Started
Run Primary Dataset.py to create an initial dataset.
Train the model using the dataset with Training.
Use Sign language detection.py to perform real-time sign language detection.
Optionally, use Dataset Updation.py to add more data for specific signs.
Requirements
Python 3.7+
OpenCV
TensorFlow
MediaPipe
NumPy, Pandas, Scikit-learn
Future Improvements
Adding more sign gestures.
Improving detection accuracy in varying lighting conditions.
Optimizing model for real-time performance on lower-end devices.
