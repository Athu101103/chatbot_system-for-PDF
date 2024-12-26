from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
import os
import requests
from utils import process_pdf, create_bm25_index

app = FastAPI()

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

documents = []
bm25 = None

# Endpoint for uploading a document
@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        content = process_pdf(file_path)
        documents.append(content)
        global bm25
        bm25 = create_bm25_index(documents)
        
        return {"message": f"File '{file.filename}' uploaded and processed successfully!"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint for querying the document
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_document(request: QueryRequest):
    try:
        if not bm25:
            return JSONResponse(content={"error": "No documents uploaded yet!"}, status_code=400)

        # Tokenize query and retrieve top documents
        query_tokens = request.query.split()
        scores = bm25.get_scores(query_tokens)
        top_docs_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        relevant_docs = [documents[i] for i in top_docs_idx]
        context = "\n".join(relevant_docs)

        # Prepare the Groq AI request payload
        user_input = f"Question: {request.query}\nContext:\n{context}"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }

        # Send the request to Groq AI
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            answer_content = result.get("choices", [{}])[0].get("message", {}).get("content", "No answer found.")
            
            return {
                "answer": answer_content,
                "sources": [documents[i] for i in top_docs_idx]
            }
        else:
            return JSONResponse(content={"error": f"Groq AI API error: {response.text}"}, status_code=response.status_code)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
