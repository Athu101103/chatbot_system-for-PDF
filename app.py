import streamlit as st
from langchain.document_loaders import PyPDFLoader
from rank_bm25 import BM25Okapi
import os
import requests  # For Groq AI API requests

# Replace with your Groq AI API key
GROQ_API_KEY = "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"

# Streamlit App
st.title("Document Chatbot System")

# Step 1: File Upload
st.header("Upload a PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.success(f"File {uploaded_file.name} uploaded successfully!")

    # Save uploaded file to a temporary location
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 2: Load PDF Content
    with st.spinner("Processing the document..."):
        pdf_loader = PyPDFLoader(temp_file_path)
        documents = pdf_loader.load()
        chunks = [doc.page_content for doc in documents]

    st.write("Document successfully processed!")

    # Step 3: Create BM25 Index for Retrieval
    with st.spinner("Building BM25 index..."):
        try:
            st.write("Creating BM25 index...")

            # Tokenize the chunks
            tokenized_chunks = [chunk.split() for chunk in chunks]
            bm25 = BM25Okapi(tokenized_chunks)

            st.success("BM25 index created successfully!")
        except Exception as e:
            st.error(f"Error in BM25 index creation: {e}")
            st.stop()

    # Step 4: Ask Questions
    st.header("Ask Questions About the Document")
    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Getting your answer..."):
                try:
                    # Tokenize user query
                    query_tokens = user_query.split()

                    # Retrieve relevant document chunks based on BM25 ranking
                    scores = bm25.get_scores(query_tokens)
                    top_docs_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                    relevant_docs = [chunks[i] for i in top_docs_idx]

                    # Concatenate the question and relevant document chunks into a single string
                    context = "\n".join(relevant_docs)  # Join relevant chunks as a string
                    user_input = f"Question: {user_query}\nContext:\n{context}"

                    # Make an API call to Groq AI with the user input as a string
                    groq_url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "llama3-8b-8192",  # Specify the model
                        "messages": [
                            {
                                "role": "user",  # Specify the role as 'user'
                                "content": user_input  # Provide the concatenated content (question + context)
                            }
                        ]
                    }

                    # Send the request to Groq API
                    response = requests.post(groq_url, json=data, headers=headers)
                    if response.status_code == 200:
                        result = response.json()

                        # Extract only the content of the response
                        answer_content = result.get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
                        
                        # Display the extracted content
                        st.write("**Answer:**")
                        st.write(answer_content)
                    else:
                        st.error(f"Error from Groq AI API: {response.text}")
                except Exception as e:
                    st.error(f"Error while querying: {e}")
        else:
            st.warning("Please enter a question.")
    
    # Cleanup: Delete the temporary file
    os.remove(temp_file_path)
