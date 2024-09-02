import cv2
import numpy as np
import time
import os

# Constants for size classification (in pixels)
SMALL_THRESHOLD = 20    # Adjust these thresholds based on calibration
MEDIUM_THRESHOLD = 40

# Conversion factor from pixels to real-world units (e.g., centimeters)
pixels_per_metric = 10  # Adjust this based on calibration

# Initialize video capture with a video file
input_video_path = '/Users/internalis/Documents/Rock_classification/videos/11sec_multiple.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(input_video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Extract the input file name without extension
input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
output_dir = os.path.join(os.path.dirname(input_video_path), 'runs')

# Create the 'runs' directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Construct the base output file path
output_video_path = os.path.join(output_dir, f"{input_filename}_output.mp4")

# If the file already exists, append a number to the filename
counter = 1
while os.path.exists(output_video_path):
    output_video_path = os.path.join(output_dir, f"{input_filename}_output_{counter}.mp4")
    counter += 1

# Flag to save the output video
save_output = False  # Set to True if you want to save the output video

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer if saving output
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use H.264 codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

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

        # Calculate size in cm
        width_cm = w / pixels_per_metric
        height_cm = h / pixels_per_metric
        size_cm = round((width_cm + height_cm) / 2)  # Average size and round to nearest cm

        if size_cm < 2 or size_cm > 10:
            continue

        size_label = f'{size_cm} cm'

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

    # Save the processed frame to the output video
    if save_output:
        out.write(frame)

    # Display the resulting frame
    cv2.imshow('Pebble Classification and Speed Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
