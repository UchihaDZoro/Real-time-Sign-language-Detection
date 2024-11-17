import cv2
import mediapipe as mp
import csv
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define signs and label dictionary
signs = ["Hello", "OK", "I love you", "See you again", "Thank you"]

# CSV file to store keypoints
csv_file = "sign_language_keypoints.csv"

# Prepare CSV file with headers
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header with landmark point names (x, y, z for each of the 21 landmarks) and label
    header = [f"x_{i}" for i in range(21)] + [f"y_{i}" for i in range(21)] + [f"z_{i}" for i in range(21)] + ["label"]
    writer.writerow(header)

# Initialize camera
cap = cv2.VideoCapture(1)

# Loop through each sign and capture 10 images
for idx, sign in enumerate(signs):
    print(f"Starting to capture images for sign: {sign}")
    
    for img_num in range(10):  # Capture 10 images per sign
        print(f"Capturing image {img_num + 1} for sign: {sign}")
        input("Press Enter to capture...")  # Wait for user input to capture

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture image.")
            continue

        # Convert the BGR image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and extract if hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Optional: Draw landmarks on the image
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])

                # Append the label to the keypoints
                keypoints.append(sign)

                # Write the keypoints and label to the CSV file
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(keypoints)

                print(f"Keypoints for image {img_num + 1} of {sign} saved to CSV.")
        else:
            print("No hand detected. Try again.")

        # Display the image (optional)
        cv2.imshow("Sign Capture", frame)
        time.sleep(2)  # Delay to avoid capturing too fast

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Keypoint data collection completed.")