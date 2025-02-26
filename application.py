from flask import Flask, request, render_template, jsonify, Response
from flask_jsglue import JSGlue
import util
import os
from werkzeug.utils import secure_filename
import time
import cv2
import helper
import settings
from pathlib import Path
import streamlit as st

application = Flask(__name__)
jsglue = JSGlue()
jsglue.init_app(application)

# Load your classification artifacts (existing logic)
util.load_artifacts()

# Load the model from your "Streamlit project" approach
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    print(f"Unable to load model. Check the specified path: {model_path}")
    print(ex)

@application.route("/")
def home():
    return render_template("home.html")

# Classify waste (existing feature)
@application.route("/classifywaste", methods=["POST"])
def classifywaste():
    image_data = request.files["file"]
    
    # Save the image to uploads folder
    basepath = os.path.dirname(__file__)
    image_path = os.path.join(basepath, "uploads", secure_filename(image_data.filename))
    image_data.save(image_path)
    
    try:
        predicted_value, details, video1, video2 = util.classify_waste(image_path)
    finally:
        time.sleep(1)  # Give time for the system to release the file
        image_data.close()
        try:
            os.remove(image_path)
        except PermissionError:
            print(f"PermissionError: Could not delete {image_path}, it might be locked by another process")
    
    # Return the classification result
    return jsonify(predicted_value=predicted_value, details=details, video1=video1, video2=video2)

# Frame generator for live video
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Classify each frame using a "play_webcam_frame"-style function
        frame, _ = helper.play_webcam_frame(frame, model)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break

        # Yield each frame in a multipart/x-mixed-replace response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

# Route that streams the webcam feed with classification overlays
@application.route("/live_video")
def live_video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    application.run()
