from flask import Flask, Response
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import os
import time
from threading import Lock

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# File paths for model files
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_folder = os.path.join(script_dir, 'assets')
classFile = os.path.join(assets_folder, 'coco.names')
configPath = os.path.join(assets_folder, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(assets_folder, 'frozen_inference_graph.pb')

# Load class labels
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Initialize the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Lock for thread safety
lock = Lock()

# Store the most recent frame
frame_data = None

# Function to process the image
def process_image(img_data):
    try:
        # Convert byte array to NumPy array and decode the image
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Object detection
        class_ids, confs, bbox = net.detect(img, confThreshold=0.6)

        # Draw bounding boxes and labels based on confidence
        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
                if confidence >= 0.7:  # Green box for high confidence
                    color = (0, 255, 0)
                    label_color = (0, 255, 0)
                elif 0.5 <= confidence < 0.7:  # Yellow box for moderate confidence
                    color = (0, 255, 255)
                    label_color = (0, 255, 255)
                else:
                    continue

                # Draw the bounding box and label
                cv2.rectangle(img, box, color=color, thickness=3)
                cv2.putText(img, classNames[class_id - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)

        # Encode image to JPEG format for live streaming
        _, img_encoded = cv2.imencode('.jpg', img)
        return img_encoded.tobytes()

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# WebSocket event to handle incoming images from the ESP32-CAM
@socketio.on('image_stream')
def handle_image_stream(img_data):
    global frame_data
    with lock:
        # Process and store the latest frame for streaming
        frame_data = process_image(img_data)
        if frame_data:
            print("Image received and processed.")
        else:
            print("Error processing image")

# Route for the video feed that streams the processed images
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    # return Response('Somthing');
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
