import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import plantcv as pcv

# Define pixels per cm based on the known scale bar length
PIXELS_PER_CM = 2817 / 6  # 469.5 pixels per cm

# Function to process image using PlantCV
def process_image(image):
    # Convert PIL image to OpenCV format
    img = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use PlantCV to threshold leaves
    mask = pcv.threshold.binary(gray, threshold=100, max_value=255, object_type="dark")

    # Fill holes and clean noise (Corrected functions)
    mask = pcv.fill(mask)  # Updated function
    mask = pcv.dilate(mask, ksize=5)  # Instead of closing

    # Find and measure leaf objects
    analysis_image, leaf_contours, leaf_hierarchy = pcv.find_objects(img, mask)
    leaf_count = len(leaf_contours)

    # Measure leaf traits and convert to cm
    leaf_data = []
    for i, contour in enumerate(leaf_contours):
        shape_data = pcv.analyze_boundaries(img, contour)
        area_cm2 = shape_data['area'] / (PIXELS_PER_CM**2)
        perimeter_cm = shape_data['perimeter'] / PIXELS_PER_CM
        width_cm = shape_data['width'] / PIXELS_PER_CM
        height_cm = shape_data['height'] / PIXELS_PER_CM

        leaf_data.append([
            i + 1,  # Leaf number
            area_cm2, 
            perimeter_cm, 
            width_cm, 
            height_cm
        ])

    return analysis_image, mask, leaf_data, leaf_count

# Streamlit UI
st.title("Potato Leaf Detection & Measurement (Accurate Scale in CM)")
st.write("Upload images for **automatic leaf segmentation, size measurement, and accuracy validation in cm.**")

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
        df = pd.DataFrame(leaf_data, columns=["Leaf #", "Leaf Area (cm²)", "Perimeter (cm)", "Width (cm)", "Height (cm)"])
        df.insert(0, "Image", uploaded_file.name)
        results_list.append(df)

    # Combine all results into a single table
    final_df = pd.concat(results_list, ignore_index=True)
    st.dataframe(final_df)

    # Download results as CSV
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "leaf_measurements_cm.csv", "text/csv")
