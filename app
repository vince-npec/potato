import streamlit as st
import cv2
import numpy as np
import os
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from PIL import Image
import pandas as pd

# Function to process image
def process_image(image):
    # Convert to OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding for segmentation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise
    binary = remove_small_objects(label(binary), min_size=500)
    binary = (binary > 0).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Compute phenotypic features
    data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        eccentricity = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        data.append([area, perimeter, eccentricity, circularity])
    
    return binary, data

# Streamlit UI
st.title("Potato Leaf Phenotyping")
st.write("Upload images for phenotyping analysis")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        binary, features = process_image(image)
        
        # Display images
        st.image(image, caption="Original Image", use_column_width=True)
        st.image(binary, caption="Segmented Image", use_column_width=True, clamp=True)
        
        # Store results
        for feature in features:
            results.append([uploaded_file.name] + feature)
    
    # Convert results to DataFrame and display
    if results:
        df = pd.DataFrame(results, columns=["Image", "Area", "Perimeter", "Eccentricity", "Circularity"])
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "phenotyping_results.csv", "text/csv")
