from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from Branched_RAG_Model import workflow
import os
import uvicorn

app = FastAPI(
    title="Knowledge Assistant API",
    description="FastAPI demo template",
    version="1.0.0"
)

UPLOAD_DIR = "Uploaded_PDF_Files/"

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "message": "PDF uploaded successfully",
        "file_name": file.filename,
        "file_path": file_path
    }


class QuickButtonRequest(BaseModel):
    button_pressed:str
    file_path:str

@app.post("/quick")
def quick_button(request: QuickButtonRequest):
    button_request=request.button_pressed
    filePath = request.file_path

    query="none"

    if button_request=="summary":
        query='''You are a professional summary creator.
                Generate a clear, concise, and professional summary of the provided data that adds meaningful value for the user.
                Focus on the key purpose, important points, and main conclusions, and ensure the summary is strictly based on the given content.
            '''

    elif button_request=="tables":
        query='''
                You are a professional data extraction expert.
                Analyze the provided document content and identify any tables or structured data present.
                If tables are found, extract them and present the information in a clean, well-formatted table with bold column headers and clearly separated rows using borders.
                Preserve the original relationships, units, and meanings of the data.
                If no tables are available in the document, clearly state: “Table is not available.
            '''

    elif button_request=="insights":
        query='''
            You are a professional document analysis expert.
            Analyze the provided document content and extract the most important insights.
            Present the insights as short, clear bullet points that highlight key findings, patterns, implications, or conclusions.
        '''
    elif button_request=="simple":
        query='''
                You are a professional explainer.
                Explain the provided document content in a simple, clear, and easy-to-understand manner for a non-technical audience.
                Use short sentences, plain language, and examples where helpful, while ensuring the explanation is strictly based on the given content.
            '''

    elif button_request=="notes":
        query='''
                You are a professional note-taking assistant.
                Convert the provided document content into clear, well-structured notes.
                Organize the information using headings, subheadings, and bullet points for easy reading and quick revision.
            '''
    
    initial_data={
        "pdf_file_path": filePath,
        "query": query
    }

    answer=workflow.invoke(initial_data)
    return {
        "query": query,
        "answer": answer['answer']
    }


class QueryRequest(BaseModel):
    query:str
    file_path:str

@app.post("/ask")
def ask_question(request: QueryRequest):
    filePath = request.file_path
    query = request.query

    initial_data={
        "pdf_file_path": filePath,
        "query": query
    }
    answer=workflow.invoke(initial_data)
    return {
        "query": query,
        "answer": answer['answer']
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )