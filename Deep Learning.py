import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape data
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define and compile the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Initialize the camera
cap = cv2.VideoCapture(0)
roi_x, roi_y, roi_width, roi_height = 200, 200, 200, 200

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the model's input shape
    resized_frame = cv2.resize(gray, (28, 28))

    # Normalize the frame
    normalized_frame = resized_frame / 255.0

    # Reshape the frame to match the model's input shape
    input_frame = normalized_frame.reshape(1, 28, 28, 1)

    # Make a prediction
    prediction = model.predict(input_frame)

    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    # Display the frame with the predicted digit
    cv2.putText(frame, f"Predicted Digit: {predicted_digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
    # Display the frame
    cv2.imshow('Handwritten Digit Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
