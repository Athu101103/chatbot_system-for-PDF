import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi

def process_pdf(file_path):
    """
    Extracts text from a PDF file.
    :param file_path: Path to the PDF file.
    :return: A single string containing the text of the PDF.
    """
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    doc.close()
    return " ".join(text)

def create_bm25_index(documents):

    tokenized_documents = [doc.split() for doc in documents]
    return BM25Okapi(tokenized_documents)
