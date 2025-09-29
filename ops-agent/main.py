import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports ---
# These are the specific tools we need from the LangChain library
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from the .env file
load_dotenv()

# --- 1. Global Variables & Models ---

# This variable will hold our searchable knowledge base.
# We make it global so it can be created at startup and used by API calls later.
vectorstore = None

# This is a Pydantic model. It defines the structure of the data we expect
# for a ticket. It's like a contract: if you send a ticket, it MUST have
# an id (integer), a title (string), and a description (string).
class Ticket(BaseModel):
    id: int
    title: str
    description: str

# --- 2. FastAPI App Initialization ---
app = FastAPI(title="OpsAgent - The On-Call Co-pilot")

# --- 3. Startup Logic: Building the Knowledge Base ---

# This is a special FastAPI "event handler". The code inside this function
# will run only ONCE, right after the application starts.
# It's the perfect place to do our heavy, one-time setup work.
@app.on_event("startup")
def startup_event():
    print("🚀 Server is starting up. Beginning knowledge base ingestion...")
    
    # Use the global keyword to modify the global 'vectorstore' variable
    global vectorstore
    
    # This path points to where our SOPs are inside the Docker container.
    # Our docker-compose.yml file mounts our local 'sops' folder to this exact location.
    sops_folder_path = "/app/sops" 

    # Step A: Load the documents
    # DirectoryLoader is a LangChain tool that finds and loads all documents in a folder.
    # We tell it to look for Markdown files (.md).
    loader = DirectoryLoader(sops_folder_path, glob="**/*.md")
    docs = loader.load()
    print(f"📚 Found and loaded {len(docs)} SOP document(s).")
    
    # Step B: Split documents into smaller chunks
    # LLMs can't process huge documents at once. We split them into smaller,
    # more manageable pieces. Overlap helps maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"📄 Split documents into {len(splits)} chunks.")
    
    # Step C: Create embeddings
    # This is the "magic" part. OpenAIEmbeddings converts our text chunks into
    # numerical representations (vectors) that capture their semantic meaning.
    # It requires our OPENAI_API_KEY to work.
    embeddings = OpenAIEmbeddings()
    
    # Step D: Create the FAISS vector store
    # FAISS is a super-fast library for searching through these numerical vectors.
    # We take our text chunks and their corresponding embeddings and store them
    # in this special database. This is our agent's "brain" or "memory".
    print("🧠 Building the FAISS vector store... (this may take a moment)")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    print("✅ Knowledge base is built and ready!")

# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    # A simple endpoint to check if the server is alive and running.
    return {"service": "OpsAgent", "status": "running"}

# We will build out the logic for this endpoint in the next phase.
# For now, it's just a placeholder that accepts a ticket.
@app.post("/tickets")
async def process_ticket(ticket: Ticket):
    print(f"Received ticket {ticket.id}: {ticket.title}")
    return {"status": "Ticket received, processing placeholder", "ticket_id": ticket.id}