import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import plantcv as pcv

# Function to process image using PlantCV
def process_image(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use PlantCV thresholding to segment leaves
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Filter out small noise and non-leaf regions
    mask = pcv.morphology.fill_holes(mask)
    mask = pcv.morphology.closing(mask, 5)

    # Find and measure leaf objects
    analysis_image, leaf_contours, leaf_hierarchy = pcv.find_objects(img, mask)
    leaf_count = len(leaf_contours)

    # Measure leaf traits
    leaf_data = []
    for i, contour in enumerate(leaf_contours):
        shape_data = pcv.morphology.analyze_boundaries(img, contour)
        leaf_data.append([
            i + 1,  # Leaf number
            shape_data['area'], 
            shape_data['perimeter'], 
            shape_data['width'], 
            shape_data['height']
        ])

    return analysis_image, mask, leaf_data, leaf_count

# Streamlit UI
st.title("Potato Leaf Detection & Phenotyping with PlantCV")
st.write("Upload images for **automatic leaf segmentation, size measurement, and accuracy validation.**")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results_list = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        analysis_image, mask, leaf_data, leaf_count = process_image(image)

        # Convert mask to PIL image
        mask_pil = Image.fromarray(mask)

        # Display original image
        st.image(image, caption="Original Image", use_container_width=True)

        # Display PlantCV segmentation mask
        st.image(mask_pil, caption="Leaf Segmentation Mask", use_container_width=True)

        # Store results in DataFrame
        df = pd.DataFrame(leaf_data, columns=["Leaf #", "Leaf Area (px²)", "Perimeter (px)", "Width (px)", "Height (px)"])
        df.insert(0, "Image", uploaded_file.name)
        results_list.append(df)

    # Combine all results into a single table
    final_df = pd.concat(results_list, ignore_index=True)
    st.dataframe(final_df)

    # Download results as CSV
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "leaf_measurements.csv", "text/csv")
