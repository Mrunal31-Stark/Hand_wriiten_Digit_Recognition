import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(_, _), (X_test, _) = mnist.load_data()

# Normalize and reshape data
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# Load the trained model
model = load_model("C:/Users/Acer/Downloads/cnn-mnist-model.h5")

# Define the region of interest (ROI) within the camera frame
roi_x, roi_y, roi_width, roi_height = 200, 200, 200, 200

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Extract the ROI from the frame
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Resize the ROI to match the model's input shape
    resized_roi = cv2.resize(gray, (28, 28))

    # Normalize the ROI
    normalized_roi = resized_roi / 255.0

    # Reshape the ROI to match the model's input shape
    input_roi = normalized_roi.reshape(1, 28, 28, 1)

    # Make a prediction
    prediction = model.predict(input_roi)

    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    # Display the ROI with the predicted digit
    cv2.putText(frame, f"Predicted Digit: {predicted_digit}", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Handwritten Digit Recognition', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
