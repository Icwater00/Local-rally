import streamlit as st
import pandas as pd
from paddleocr import PaddleOCRVL
from PIL import Image
import io

# Page Config
st.set_page_config(page_title="Guild Rally OCR", page_icon="🛡️")
st.title("🛡️ Guild Rally Name Extractor")

# Load OCR Model (Cached to prevent reloading on every click)
@st.cache_resource
def load_model():
    return PaddleOCRVL()

ocr = load_model()

# UI: File Uploader
uploaded_file = st.file_uploader("Upload Rally Box Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Show the image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_container_width=True)
    
    if st.button("Extract Names"):
        with st.spinner('Processing...'):
            # Convert uploaded file to bytes for the OCR engine
            img_bytes = uploaded_file.getvalue()
            
            # Run the OCR Prediction
            results = ocr.predict(img_bytes)
            
            # Extract and clean names
            all_text = []
            for res in results:
                all_text.extend([line.strip() for line in res.rec_text if len(line.strip()) > 2])
            
            # Remove duplicates (useful if names appear twice in UI)
            unique_names = list(dict.fromkeys(all_text))
            
            # Create Table
            df = pd.DataFrame(unique_names, columns=["Character Name"])
            
            st.success(f"Successfully extracted {len(df)} names!")
            st.write(df)

            # Generate CSV for download
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name="rally_members.csv",
                mime="text/csv",
            )