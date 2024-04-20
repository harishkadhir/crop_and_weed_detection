import cv2
from ultralytics import YOLO

# Initialize the YOLOv5 model with the specified weights file
model = YOLO("E:/Colab Notebooks/Best.pt")  # Using forward slashes for file path

# Define the video source (webcam or video file)
video_source = "0"  # Default webcam source

# Open the video source
cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)

# Check if the video source is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video source")
    exit()

# Define class names for mapping numerical labels
class_names = {0: "crop", 1: "weed"}

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Break the loop if the frame is not captured successfully
    if not ret:
        break

    # Perform object detection on the frame
    results = model.predict(frame)

    # Iterate over each element in the results list
    for result in results:
        # Check if there are detections
        if result.boxes is not None:
            # Draw bounding boxes and labels on the frame
            for box, label in zip(result.boxes.xyxy, result.names):
                x1, y1, x2, y2 = map(int, box)
                label_name = class_names[int(label)]  # Map numerical label to class name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw bounding box
                cv2.putText(frame, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Write label

    # Display the output frame
    cv2.imshow("Object Detection", frame)

    # Wait for a short time and check for user input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video source
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()





