import streamlit as st
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Load YOLOv8 model (Replace 'yolov8n.pt' with a fine-tuned model if available)
model = YOLO("yolov8n.pt")

# Function to process image and detect leaves
def process_image(image):
    # Convert PIL Image to NumPy array
    img = np.array(image)

    # Perform YOLO inference
    results = model(img)

    # Extract detected objects (leaves)
    leaf_count = 0
    for result in results:
        for box in result.boxes:
            leaf_count += 1  # Count the number of detected leaves

    return leaf_count

# Streamlit UI
st.title("Potato Leaf Detection and Counting with YOLOv8")
st.write("Upload images for automatic leaf detection and counting")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        leaf_count = process_image(image)

        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        st.write(f"**Detected Leaves:** {leaf_count}")

        # Store results
        results.append([uploaded_file.name, leaf_count])

    # Convert results to DataFrame and display
    df = pd.DataFrame(results, columns=["Image", "Leaf Count"])
    st.dataframe(df)

    # Download results as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "leaf_count_results.csv", "text/csv")
