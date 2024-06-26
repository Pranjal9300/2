import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import layoutparser as lp
import pytesseract

# Directly include the punkt data
punkt_data = """
<...>  # Paste the content of the punkt data file here
"""

# Initialize the tokenizer
tokenizer = PunktSentenceTokenizer(punkt_data)

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

# Load the summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

# Function to summarize text
def summarize_text(text, summarizer, max_chunk_size=1000):
    sentences = tokenizer.tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)

# Title of the app
st.title("PDF Summarizer AI")

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

        # User options
        options = st.multiselect(
            "Choose what you want to do:",
            ["Extract Headings", "Summarize Text"]
        )

        if "Extract Headings" in options and pdf_text:
            # Display extracted headings with page numbers
            st.subheader("Extracted Headings")
            for heading, page_num in headings:
                st.write(f"{heading} (Page {page_num})")

        if "Summarize Text" in options and pdf_text:
            # Summarize the extracted text
            summarizer = load_summarizer()
            summary = summarize_text(pdf_text, summarizer)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)
