import streamlit as st
import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Load YOLOv8 model for leaf detection
model = YOLO("yolov8n.pt")  # Replace with a fine-tuned model for leaves if available

# Function to process image
def process_image(image):
    # Convert to OpenCV format
    img = np.array(image)
    
    # Perform YOLO inference
    results = model(img)
    
    # Extract detected objects
    leaf_count = 0
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Adjust class ID if needed
                leaf_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img, leaf_count

# Streamlit UI
st.title("Potato Leaf Phenotyping with YOLOv8")
st.write("Upload images for leaf detection and counting")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        processed_img, leaf_count = process_image(image)
        
        # Display images
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(processed_img, caption=f"Processed Image (Leaf Count: {leaf_count})", use_column_width=True)
        
        # Store results
        results.append([uploaded_file.name, leaf_count])
    
    # Convert results to DataFrame and display
    df = pd.DataFrame(results, columns=["Image", "Leaf Count"])
    st.dataframe(df)

    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "leaf_count_results.csv", "text/csv")
