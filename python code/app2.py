from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import numpy as np
import requests
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load YOLO model
try:
    model_path = r"C:\\Users\\KOTLA LATHISH\\Desktop\\AWP AND MLUP PROJECT\\Helmet-Detection-in-traffic-main\\PycharmProjects\\PythonProject\\runs\\detect\\helmet_detection10\\weights\\best.pt"
    model = YOLO(model_path)
    print("Local YOLO Model Loaded Successfully!")
    print("Model Classes:", model.names)
except Exception as e:
    print(f"Error loading local model: {e}")
    model = None

# Roboflow API Configuration (Free Tier)
ROBOFLOW_API_KEY = "7nWxplYoZjJ1vtr98d0z"  # Replace with your actual key
ROBOFLOW_MODEL_ID = "hard-hat-workers"  # Pre-trained helmet detection model
ROBOFLOW_VERSION = "1"
ROBOFLOW_ENDPOINT = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}/{ROBOFLOW_VERSION}"


def detect_with_roboflow(image_path):
    """Use Roboflow API for high-accuracy detection"""
    try:
        with open(image_path, 'rb') as f:
            response = requests.post(
                ROBOFLOW_ENDPOINT,
                files={"file": f},
                data={"api_key": ROBOFLOW_API_KEY}
            )

        if response.status_code == 200:
            predictions = response.json()['predictions']
            return predictions
        else:
            print(f"Roboflow API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling Roboflow API: {e}")
        return None


def draw_roboflow_results(image, predictions):
    """Draw bounding boxes from Roboflow API results"""
    helmet_count = 0
    no_helmet_count = 0

    for pred in predictions:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        w = int(pred['width'])
        h = int(pred['height'])
        class_name = pred['class']
        confidence = pred['confidence']

        if class_name.lower() in ['helmet', 'with_helmet', 'hardhat']:
            color = (0, 255, 0)  # Green
            label = "Helmet"
            helmet_count += 1
        else:
            color = (0, 0, 255)  # Red
            label = "No Helmet"
            no_helmet_count += 1

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image, helmet_count, no_helmet_count


def detect_objects_in_image(image_path):
    """Hybrid detection using both local model and Roboflow API"""
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read image!")
        return None, 0, 0

    # First try with local model
    local_helmet_count = 0
    local_no_helmet_count = 0

    if model is not None:
        results = model.predict(image, save=False, conf=0.5)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name.lower() in ['helmet', 'with helmet']:
                    color = (0, 255, 0)  # Green
                    label = "Helmet (Local)"
                    local_helmet_count += 1
                elif class_name.lower() in ['no helmet', 'without helmet']:
                    color = (0, 0, 255)  # Red
                    label = "No Helmet (Local)"
                    local_no_helmet_count += 1

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Fallback to Roboflow API if local model has low confidence or no detections
    if local_helmet_count + local_no_helmet_count < 1:
        print("Using Roboflow API for better accuracy...")
        predictions = detect_with_roboflow(image_path)
        if predictions:
            image, api_helmet_count, api_no_helmet_count = draw_roboflow_results(image, predictions)
            helmet_count = local_helmet_count + api_helmet_count
            no_helmet_count = local_no_helmet_count + api_no_helmet_count
        else:
            helmet_count = local_helmet_count
            no_helmet_count = local_no_helmet_count
    else:
        helmet_count = local_helmet_count
        no_helmet_count = local_no_helmet_count

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detected_{int(time.time())}.jpg")
    cv2.imwrite(output_path, image)
    return output_path, helmet_count, no_helmet_count


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        output_path, helmet_count, no_helmet_count = detect_objects_in_image(image_path)
        if output_path:
            return render_template('index.html',
                                   image_output=output_path,
                                   helmet_count=helmet_count,
                                   no_helmet_count=no_helmet_count)
        else:
            return "Error processing image", 500


def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # For real-time, we'll use only the local model for speed
        if model is not None:
            results = model.predict(frame, save=False, conf=0.5)
            helmet_count = 0
            no_helmet_count = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]

                    if class_name.lower() in ['helmet', 'with helmet']:
                        color = (0, 255, 0)  # Green
                        label = "Helmet"
                        helmet_count += 1
                    elif class_name.lower() in ['no helmet', 'without helmet']:
                        color = (0, 0, 255)  # Red
                        label = "No Helmet"
                        no_helmet_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display counts on the frame
            cv2.putText(frame, f"Helmets: {helmet_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"No Helmets: {no_helmet_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/realtime')
def realtime():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)