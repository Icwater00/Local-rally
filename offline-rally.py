import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="PaddleOCR App", layout="centered")
st.title("📝 Image to Text with PaddleOCR")
st.write("Upload an image, and the AI will extract the text for you.")

# 2. Cache the OCR model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    # det=True (detect text), rec=True (recognize text)
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_model()

# 3. File Uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a button to trigger extraction
    if st.button("Extract Text"):
        with st.spinner("Analyzing image..."):
            
            # Convert PIL image to a NumPy array (which PaddleOCR requires)
            image_np = np.array(image)
            
            # Run the OCR
            result = ocr.ocr(image_np, cls=True)
            
            # 4. Process and Display Results
            if not result or result[0] is None:
                st.warning("No text detected in this image.")
            else:
                st.success("Extraction Complete!")
                
                # Create lists to hold our formatted data
                extracted_data = []
                full_text = ""
                
                # result[0] contains the actual lines found in the image
                for line in result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    full_text += text + "\n"
                    extracted_data.append({"Text": text, "Confidence": f"{confidence:.2%}"})
                
                # Display as a clean table
                st.subheader("Extracted Data Table")
                st.table(extracted_data)
                
                # Provide a text area for easy copying
                st.subheader("Raw Text (Copy/Paste)")
                st.text_area("Copied Text", full_text, height=200)
