from ultralytics import YOLO
import time
import streamlit as st
import cv2
import settings
import threading

def sleep_and_clear_success():
    time.sleep(3)
    st.session_state['recyclable_placeholder'].empty()
    st.session_state['non_recyclable_placeholder'].empty()
    st.session_state['hazardous_placeholder'].empty()

def load_model(model_path):
    model = YOLO(model_path)
    return model

def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)
    
    return recyclable_items, non_recyclable_items, hazardous_items

def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")

def _display_detected_frames(model, image):
    """Process the frame, draw bounding boxes with labels (name + confidence),
    overlay waste classification summary, and return the annotated image."""
    
    # Resize image for consistent processing
    image = cv2.resize(image, (640, int(640 * (9 / 16))))
    
    # Run the YOLO model prediction
    res = model.predict(image, conf=0.6)
    names = model.names  # class names provided by the model

    # Get the default annotated image (if YOLO's plot() already draws boxes)
    annotated_frame = res[0].plot()

    # Iterate over each detected bounding box to overlay custom label info
    for box in res[0].boxes:
        # Extract bounding box coordinates.
        # YOLO's box.xyxy is typically a 2D array (or tensor) with shape (1, 4)
        coords = box.xyxy[0]
        x1, y1, x2, y2 = map(int, coords)
        
        # Retrieve the class id, name, and confidence for the detection
        class_id = int(box.cls)
        class_name = names[class_id]
        # Depending on your YOLO version, box.conf might be a tensor or list;
        # we convert it to a float and multiply by 100 for percentage.
        confidence = float(box.conf) * 100  
        label = f"{class_name}: {confidence:.1f}%"
        
        # (Re)draw the bounding box (in case the default plot doesn't include it)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Gather detected items for waste classification
    detected_items = set([names[int(box.cls)] for box in res[0].boxes])
    recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

    # Overlay classification summary on the frame
    y_offset = 30  # starting y position for text overlay
    for category, items in [("Recyclable", recyclable_items),
                            ("Non-Recyclable", non_recyclable_items),
                            ("Hazardous", hazardous_items)]:
        if items:
            # Create a string of items after removing dashes from names
            text = f"{category}: {', '.join(remove_dash_from_class_name(item) for item in items)}"
            cv2.putText(annotated_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25

    return annotated_frame



def play_webcam_frame(frame, model):
    # Run the prediction
    results = model.predict(frame, conf=0.6)
    
    # Get the annotated frame
    annotated_frame = results[0].plot()
    
    # Extract detected items
    detected_items = set([model.names[int(c)] for c in results[0].boxes.cls])
    
    # Return the annotated frame and detected items
    return annotated_frame, detected_items


def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH
    if st.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model,st_frame,image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))