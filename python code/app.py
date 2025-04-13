import cv2
from ultralytics import YOLO

# Load the YOLOv8 model with the best weights
model = YOLO('C:\\Users\\KOTLA LATHISH\\Desktop\\AWP AND MLUP PROJECT\\Helmet-Detection-in-traffic-main\\PycharmProjects\\PythonProject\\runs\\detect\\helmet_detection10\\weights\\best.pt')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    # Perform object detection on the frame
    results = model.predict(source=frame, save=False, conf=0.5)

    # Iterate through the detected results
    for result in results[0].boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        cls = int(result.cls[0])

        # Convert the class index to a label
        class_name = model.names[cls]

        # Set the color and label based on the class
        if class_name.lower() == 'with helmet':
            color = (0, 255, 0)  # Green for wearing helmet
            label = "Helmet Detected"
        elif class_name.lower() == 'without helmet':
            color = (0, 0, 255)  # Red for not wearing helmet
            label = "No Helmet"
        else:
            color = (255, 255, 255)  # White for unknown labels
            label = "Unknown"

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow('Helmet Detection', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()