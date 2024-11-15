import os
import cv2
import numpy as np
import eventlet
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, PlainTextResponse
from threading import Lock

# Initialize FastAPI app
app = FastAPI()

# File paths for model files
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_folder = os.path.join(script_dir, 'assets')
classFile = os.path.join(assets_folder, 'coco.names')
configPath = os.path.join(assets_folder, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(assets_folder, 'frozen_inference_graph.pb')

# Load class labels
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Initialize the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Shared frame buffer and queue to store frames
frame_queue = []
frame_lock = Lock()

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

# WebSocket endpoint to handle incoming images from the ESP32-CAM
@app.websocket("/ws/image_stream")
async def handle_image_stream(websocket: WebSocket):
    global frame_queue
    await websocket.accept()
    try:
        while True:
            # Receive the image data (binary)
            img_data = await websocket.receive_bytes()

            # Process the image synchronously
            processed_frame = process_image(img_data)
            if processed_frame:
                with frame_lock:
                    frame_queue.append(processed_frame)  # Add processed frame to queue

    except WebSocketDisconnect as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Route to stream video feed
@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            try:
                with frame_lock:
                    if frame_queue:
                        # Get the oldest frame (first-in, first-out)
                        frame_to_send = frame_queue.pop(0)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            except Exception as e:
                print(f"Error in video stream: {e}")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# Welcome route
@app.get("/")
def go():
    return PlainTextResponse("Welcome to the FastAPI server!")

# Run FastAPI server using Uvicorn (FastAPI's ASGI server)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
