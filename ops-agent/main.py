import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports (FOR OPENAI) ---
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Global Variables & Models ---
vectorstore = None

class Ticket(BaseModel):
    id: int
    title: str
    description: str

# --- FastAPI App Initialization ---
app = FastAPI(title="OpsAgent - Powered by OpenAI")

# --- Startup Logic: Building the Knowledge Base ---
@app.on_event("startup")
def startup_event():
    print("🚀 Server is starting up. Beginning knowledge base ingestion...")
    global vectorstore
    
    sops_folder_path = "/app/sops" 

    loader = DirectoryLoader(sops_folder_path, glob="**/*.md")
    docs = loader.load()
    print(f"📚 Found and loaded {len(docs)} SOP document(s).")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"📄 Split documents into {len(splits)} chunks.")
    
    # This now uses the correct OpenAI class
    embeddings = OpenAIEmbeddings()
    
    print("🧠 Building the FAISS vector store with OpenAI embeddings... (this may take a moment)")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    print("✅ Knowledge base is built and ready!")

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"service": "OpsAgent", "status": "running", "model_provider": "OpenAI"}

@app.post("/tickets")
async def process_ticket(ticket: Ticket):
    print(f"Received ticket {ticket.id}: {ticket.title}")
    return {"status": "Ticket received, processing placeholder", "ticket_id": ticket.id}