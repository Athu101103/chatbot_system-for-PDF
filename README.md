# Document Chatbot System with Groq AI Integration

## Overview
The Document Chatbot System simplifies interactions with uploaded PDF documents by enabling users to query the content using natural language. With a FastAPI backend and a Streamlit-based frontend, the system incorporates Groq AI to provide accurate and context-aware responses.

## Key Features

### Document Upload and Processing
- Supports uploading and processing *multiple PDF documents*.
- Extracts text content from PDFs.
- Indexes text using the *BM25 algorithm* for efficient query matching.

### User-Friendly Frontend
- Interactive interface built with *Streamlit*.
- Enables users to upload multiple files and ask questions seamlessly.

### Error Handling and Reliability
- Implements robust error-handling mechanisms, including:
  - Exception handling in API endpoints.
  - Validation of user inputs.
  - Fallback mechanisms to maintain functionality when Groq AI is unavailable.
- Provides clear error messages for issues such as file upload failures or processing errors.

### Avoiding Hallucinations
- Ensures responses are rooted in the provided document content.
- Groq AI integration delivers concise answers while avoiding irrelevant or misleading information.

---

## Technology Stack

### Backend
- *FastAPI*: For API endpoints.
- *BM25*: For text indexing and retrieval.

### Frontend
- *Streamlit*: Provides an intuitive and user-friendly interface.

### AI Integration
- *Groq AI*: For enhanced natural language understanding and precise query responses.

### Libraries
- *PyMuPDF*: For PDF processing.
- *Requests*: For API communication.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Steps

1. Create and activate a virtual environment:
   bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Start the backend server:
   bash
   uvicorn app:app --reload
   

4. Run the frontend application:
   bash
   streamlit run frontend.py
   

---

## Usage

1. Open the Streamlit app in your browser (usually available at http://localhost:8501).
2. Upload one or more PDF documents.
3. Enter your question in the query input field.
4. View the AI-generated answer based on your uploaded documents.

