import streamlit as st
import cv2
import numpy as np
from skimage import measure, color, io

def calculate_leaf_area(image, scale_value):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary image
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Label connected regions of the binary image
    labeled = measure. label(binary)
    regions = measure.regionprops(labeled)

    # Calculate leaf area based on the scale
    areas = [region.area * scale_value for region in regions]
    return areas

def main():
    st.title('Potato Plant Leaf Area Measurement')
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, channels="BGR")
        
        scale_value = st.number_input("Enter scale value (e.g., mm per pixel):", min_value=0.0)
        
        if st.button('Calculate Leaf Area'):
            areas = calculate_leaf_area(image, scale_value)
            st.write(f"Leaf Areas: {areas}")

if __name__ == '__main__':
    main()
