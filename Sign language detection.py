import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("sign_language_classification_model.h5")

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Defining the label mapping based on the order used during training
label_encoder = LabelEncoder()
signs = ["Hello", "OK", "I love you", "See you again", "Thank you"]
label_encoder.fit(signs)

# Starting the camera
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame (our chocice, it will work fine even without it)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks as flattened array for prediction
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])

            # Convert keypoints to numpy array and reshape for prediction
            keypoints = np.array(keypoints).reshape(1, -1)

            # Predicting the sign
            predictions = model.predict(keypoints)
            predicted_label = np.argmax(predictions, axis=1)
            sign_name = label_encoder.inverse_transform(predicted_label)[0]

            # Displaying the predicted sign on the top left corner of the frame
            cv2.putText(frame, f"Sign: {sign_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
