from ultralytics import YOLO

# Step 1: Specify the path to your data.yaml file
data_yaml = "data.yaml"  # Update this path as needed

# Step 2: Initialize the YOLOv8 model
# Options include yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (nano, small, medium, large, extra-large)
model = YOLO('yolov8n.pt')  # Using the nano model as a base

# Step 3: Start training the model
model.train(
    data=data_yaml,
    epochs=10,    # Adjust the number of epochs as needed
    batch=10,      # Adjust batch size according to your GPU capacity
    imgsz=640,    # Image size, can be adjusted (e.g., 416, 640)
    name='helmet_detection'  # Name for the run, it will create a folder for logs and weights
)