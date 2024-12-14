import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from nltk.tokenize import sent_tokenize
import nltk
from Summarization.summarization import summarize_text
from QA.qa_system import process_questions
from nltk.data import find
nltk.download('punkt')

try:
    find('tokenizers/punkt_tab.zip')
    print("punkt_tab resource already downloaded.")
except LookupError:
    nltk.download('punkt_tab')


# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Read from stream
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to format text into bullet points
def format_as_bullets(text, num_points=5):
    """Convert the text into bullet points by extracting sentences."""
    sentences = sent_tokenize(text)  # Split text into sentences
    return [f"- {sentence.strip()}" for sentence in sentences[:num_points]]

# Streamlit Sidebar
st.sidebar.title("Options")
menu_option = st.sidebar.radio("Choose an option:", ["Summarization", "QA System", "About"])

if menu_option == "Summarization":
    # Summarization Section
    st.title("📝Text Summarization")
    st.write("Upload a PDF file or enter text to summarize it and view the top 5 bullet points.")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Text area for manual input
    user_text = st.text_area("Or, enter your text here:")

    # Summarization button
    if st.button("Summarize"):
        if uploaded_file:
            st.write("Extracting text from PDF...")
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.write("Summarizing extracted text...")
            summary = summarize_text(pdf_text)
            bullet_points = format_as_bullets(summary)
            st.subheader("Summary of PDF Content (Top 5 Points)")
            for bullet in bullet_points:
                st.write(bullet)
        elif user_text.strip():
            st.write("Summarizing entered text...")
            summary = summarize_text(user_text)
            bullet_points = format_as_bullets(summary)
            st.subheader("Summary of Entered Text (Top 5 Points)")
            for bullet in bullet_points:
                st.write(bullet)
        else:
            st.warning("Please upload a PDF or enter text for summarization.")

elif menu_option == "QA System":
    # Question Answering System Section
    st.title("Question Answering System")
    st.write("Enter a context or upload a PDF to extract context, then ask questions to get precise answers.")

    # PDF uploader for context
    uploaded_pdf = st.file_uploader("Upload a PDF file for context (optional):", type=["pdf"])

    # Context text area
    context = st.text_area("Or, enter context text manually:")

    # Extract text from uploaded PDF
    if uploaded_pdf:
        st.write("Extracting text from uploaded PDF...")
        extracted_text = extract_text_from_pdf(uploaded_pdf)
        context = extracted_text  # Overwrite the context with extracted PDF text
        st.write("Extracted Context:")
        st.write(context[:1000])  # Display a preview of the extracted text (limit to 1000 characters)

    # Questions input (multi-line for multiple questions)
    questions_input = st.text_area("Enter questions (one per line):")

    # QA button
    if st.button("Get Answers"):
        if context.strip() and questions_input.strip():
            # Process questions
            questions_list = questions_input.splitlines()  # Split by line for multiple questions
            answers = process_questions(questions_list, context)
            st.subheader("Answers")
            for question, answer in answers:
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")
                st.write("---")
        else:
            st.warning("Please provide both context and questions.")

elif menu_option == "About":
    # About Section
    st.title("About")
    st.write("""
    **Intelligent Text Assistant** is a powerful application built using Streamlit. 
    It offers multiple functionalities:

    1. **Summarization**: Upload a PDF file or enter text to generate concise summaries.
    2. **QA System**: Enter a context and ask questions to get precise answers.

    Built using state-of-the-art NLP models and libraries.
    """)
