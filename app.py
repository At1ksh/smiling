import cv2
import streamlit as st
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import requests

# === CONFIG ===
model_url="https://drive.google.com/uc?export=download&id=1JtUOuZrTLrF0lh6fulSQirwfguxAsKwx"
yolo_model_path = r"best.pt"
model_path = r"SmileLine_AutOrHse.h5"
if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

yolo_model = YOLO(yolo_model_path)
keras_model = tf.keras.models.load_model(model_path)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def process_image(image_file):
    # Convert to RGB and save temporarily
    img_rgb = Image.open(image_file).convert('RGB')
    img_np_rgb = np.array(img_rgb)

    # Convert to grayscale
    img_bw = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2GRAY)
    tmp_bw_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_bw_path, img_bw)

    # === YOLO OBB INFERENCE ===
    results = yolo_model(tmp_bw_path)
    for result in results:
        if not result.obb or result.obb.conf is None or len(result.obb.conf) == 0:
            st.error("No object detected.")
            return None

        confs = result.obb.conf.cpu().numpy()
        max_idx = confs.argmax()
        obb = result.obb.xyxyxyxy[max_idx].cpu().numpy().astype(int)
        crop_coords = obb.reshape(4, 2)

        # === PERSPECTIVE TRANSFORM ===
        ordered_pts = order_points(crop_coords.astype("float32"))
        (tl, tr, br, bl) = ordered_pts
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(img_np_rgb, M, (maxWidth, maxHeight))

        return warped

    return None

def classify_image(img):
    resized = cv2.resize(img, (256, 256))
    input_tensor = np.expand_dims(resized/255, axis=0)
    prediction = keras_model.predict(input_tensor)[0]
    return prediction

# === STREAMLIT UI ===
st.title("Smile Line Detector")
image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if image_file:
    st.image(image_file, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Processing...'):
        cropped_img = process_image(image_file)
        if cropped_img is not None:
            st.image(cropped_img, caption='Straightened Crop', use_column_width=True)
            prediction = classify_image(cropped_img)
            if(prediction<0.5):
                st.success("Atlas Nickel detected, Autobiography Car")
            else:
                st.success("Atlas Detected, HSE Car")
        else:
            st.error("No crop could be obtained from image.")
