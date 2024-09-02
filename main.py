import cv2
import numpy as np
import time

# Constants for size classification (in pixels)
SMALL_THRESHOLD = 20    # Adjust these thresholds based on calibration
MEDIUM_THRESHOLD = 40

# Conversion factor from pixels to real-world units (e.g., centimeters)
pixels_per_metric = 10  # Adjust this based on calibration

# Initialize video capture (0 for default camera or provide video file path)
cap = cv2.VideoCapture('/Users/internalis/Documents/Rock_classification/videos/singleRock.mp4')

# Dictionary to store pebble information across frames
pebbles = {}
pebble_id = 0

# Get the initial time
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistency (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to detect pebbles
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_pebbles = []

    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 50:
            continue

        # Compute bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        center = (int(x + w / 2), int(y + h / 2))

        # Calculate size (e.g., area)
        area = cv2.contourArea(contour)
        size_label = 'Small'
        if area > MEDIUM_THRESHOLD:
            size_label = 'Large'
        elif area > SMALL_THRESHOLD:
            size_label = 'Medium'

        # Attempt to match current contour with existing pebbles
        matched = False
        for pid in list(pebbles.keys()):
            prev_center = pebbles[pid]['positions'][-1]
            distance = np.linalg.norm(np.array(center) - np.array(prev_center))

            # If distance is small enough, consider it the same pebble
            if distance < 50:
                time_diff = time.time() - pebbles[pid]['last_updated']
                speed = distance / time_diff / pixels_per_metric  # units per second

                pebbles[pid]['positions'].append(center)
                pebbles[pid]['last_updated'] = time.time()
                pebbles[pid]['speed'] = speed
                pebbles[pid]['size'] = size_label

                matched = True

                # Draw annotations
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {pid}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Size: {size_label}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f'Speed: {speed:.2f} units/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break

        # If no match found, register new pebble
        if not matched:
            pebbles[pebble_id] = {
                'positions': [center],
                'last_updated': time.time(),
                'speed': 0,
                'size': size_label
            }

            # Draw annotations
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {pebble_id}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Size: {size_label}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Speed: 0 units/s', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            pebble_id += 1

    # Display the resulting frame
    cv2.imshow('Pebble Classification and Speed Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
