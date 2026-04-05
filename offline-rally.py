import streamlit as st
import pandas as pd
from paddleocr import PaddleOCRVL
from PIL import Image
import io

# 1. Page Configuration
st.set_page_config(page_title="Guild Rally OCR", page_icon="🛡️")
st.title("🛡️ Guild Rally Name Extractor")
st.markdown("Upload your **Rally Box** screenshot below to generate a list of names.")

# 2. Load OCR Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    # PaddleOCRVL infers hardware from the installed paddlepaddle package.
    return PaddleOCRVL()

try:
    ocr = load_model()
except Exception as e:
    st.error(f"Error loading OCR model: {e}")
    st.stop()

# 3. User Interface: File Uploader
uploaded_file = st.file_uploader("Upload Rally Box Screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the uploaded image for confirmation
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Screenshot', use_container_width=True)
    
    # 4. Trigger Extraction
    if st.button("Extract Names"):
        with st.spinner('Reading screenshot... this may take a moment on CPU.'):
            try:
                # Convert uploaded file to bytes for the OCR engine
                img_bytes = uploaded_file.getvalue()
                
                # Run the OCR Prediction
                results = ocr.predict(img_bytes)
                
                # 5. Data Cleaning Logic
                # We filter out very short strings (noise) and strip whitespace
                extracted_list = []
                for res in results:
                    # In PaddleOCR-VL 1.5/3.x, rec_text contains the identified strings
                    extracted_list.extend([line.strip() for line in res.rec_text if len(line.strip()) > 2])
                
                # Remove duplicates while preserving the order seen in the image
                unique_names = list(dict.fromkeys(extracted_list))
                
                if unique_names:
                    # Create the DataFrame
                    df = pd.DataFrame(unique_names, columns=["Character Name"])
                    
                    st.success(f"Successfully extracted {len(df)} names!")
                    
                    # Display table in the app
                    st.dataframe(df, use_container_width=True)

                    # 6. Generate CSV for download
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Names as CSV",
                        data=csv_data,
                        file_name="rally_members.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No names detected. Please ensure the screenshot is clear.")

            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")

# 7. Instructions for your Guildmates
with st.expander("How to use this tool"):
    st.write("""
    1. Take a screenshot of the **Rally Box** in-game.
    2. Upload the file here.
    3. Click **Extract Names**.
    4. Download the CSV and upload it to the guild spreadsheet.
    """)
