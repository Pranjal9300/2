import streamlit as st
import fitz  # PyMuPDF
import layoutparser as lp
import pytesseract
import numpy as np

# Function to extract text and headings from a range of pages in a PDF file using layoutparser
def extract_text_and_headings_from_pdf(doc, start_page, end_page):
    text = ""
    headings = []
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

    for page_number in range(start_page, end_page + 1):
        if page_number < 1 or page_number > len(doc):
            continue  # skip invalid page numbers
        page = doc[page_number - 1]
        pix = page.get_pixmap()
        image = np.array(pix.samples).reshape(pix.height, pix.width, pix.n)

        layout = model.detect(image)

        # Sort the layout elements by y-coordinate to process them top-down
        layout = lp.Layout([b for b in layout if b.type in ['Title', 'Heading', 'Text']], sort_by="y")

        for block in layout:
            block_text = pytesseract.image_to_string(block.image)
            if block.type in ['Title', 'Heading']:
                headings.append((block_text.strip(), page_number))
            text += block_text.strip() + " "
    
    return text, headings

# Title of the app
st.title("PDF Heading Detection")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    num_pages = len(doc)
    
    # Input for start and end page numbers
    start_page = st.number_input("Enter the start page number", min_value=1, value=1, step=1)
    end_page = st.number_input("Enter the end page number", min_value=1, value=min(num_pages, start_page + 9), step=1)

    if start_page > end_page:
        st.error("End page must be greater than or equal to start page")
    else:
        # Extract text and headings from the specified range of pages of the PDF
        pdf_text, headings = extract_text_and_headings_from_pdf(doc, start_page, end_page)

        # Display extracted headings with page numbers
        st.subheader("Extracted Headings")
        for heading, page_num in headings:
            st.write(f"{heading} (Page {page_num})")
