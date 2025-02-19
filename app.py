import streamlit as st
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd
import cv2

# Load YOLOv8 model (Use a fine-tuned model if available)
model = YOLO("yolov8n.pt")  # Replace with "best.pt" if you have a fine-tuned model

# Function to process the image and extract leaf properties
def process_image(image):
    # Convert PIL Image to NumPy array
    img = np.array(image)

    # Perform YOLO inference
    results = model(img)

    # Initialize mask and results list
    mask = np.zeros_like(img[:, :, 0])  # Create a blank grayscale mask
    leaf_data = []

    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            leaf_area = (x2 - x1) * (y2 - y1)
            perimeter = 2 * ((x2 - x1) + (y2 - y1))
            width, height = x2 - x1, y2 - y1

            # Draw the leaf on the mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)  # White leaf mask

            leaf_data.append([x1, y1, x2, y2, leaf_area, perimeter, width, height])

    return results, mask, leaf_data

# Streamlit UI
st.title("Potato Leaf Detection & Phenotyping with YOLOv8")
st.write("Upload images for **automatic leaf detection, size measurement, and segmentation mask generation.**")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        results, mask, leaf_data = process_image(image)

        # Convert mask to PIL image
        mask_pil = Image.fromarray(mask)

        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Display segmentation mask
        st.image(mask_pil, caption="Leaf Segmentation Mask", use_column_width=True)

        # Store results in DataFrame
        df = pd.DataFrame(leaf_data, columns=["x1", "y1", "x2", "y2", "Leaf Area (px²)", "Perimeter (px)", "Width (px)", "Height (px)"])
        df.insert(0, "Image", uploaded_file.name)
        results_list.append(df)

    # Combine all results into a single table
    final_df = pd.concat(results_list, ignore_index=True)
    st.dataframe(final_df)

    # Download results as CSV
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "leaf_measurements.csv", "text/csv")
