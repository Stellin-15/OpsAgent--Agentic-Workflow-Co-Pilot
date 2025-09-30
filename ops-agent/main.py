import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. Global Variables & Models ---
rag_chain = None

# --- NEW: In-Memory "Database" for Pending Drafts ---
# This is a simple dictionary that will act as temporary storage.
# The key will be the ticket_id, and the value will be the drafted reply.
PENDING_DRAFTS = {}

class Ticket(BaseModel):
    id: int
    title: str
    description: str

# --- 2. FastAPI App Initialization ---
app = FastAPI(title="OpsAgent - Powered by Gemini")

# --- 3. Startup Logic (No changes here) ---
@app.on_event("startup")
def startup_event():
    print("🚀 Server is starting up...")
    global rag_chain
    sops_folder_path = "/app/sops" 
    loader = DirectoryLoader(sops_folder_path, glob="**/*.md")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("✅ Knowledge base built.")
    print("🔗 Building the RAG chain...")
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert IT Operations support agent. Your goal is to draft a helpful,
        concise, and friendly Slack-style reply to a user's ticket.
        Use the following CONTEXT from our Standard Operating Procedures (SOPs) to answer.
        Do not use any other information. If the context is not relevant, say you cannot find a solution.
        CONTEXT:
        {context}
        TICKET:
        {ticket_query}
        DRAFT REPLY:
        """
    )
    rag_chain = (
        {"context": retriever, "ticket_query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain is ready!")


# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"service": "OpsAgent", "status": "running", "model_provider": "Google Gemini"}

# --- MODIFIED: The /tickets endpoint NO LONGER waits ---
@app.post("/tickets")
async def process_ticket(ticket: Ticket):
    query = f"Title: {ticket.title}\nDescription: {ticket.description}"
    
    print(f"\n🔎 Received query for ticket {ticket.id}: {query}")
    print("🤔 Thinking...")

    draft_reply = rag_chain.invoke(query)

    # Store the draft in our in-memory "database"
    PENDING_DRAFTS[ticket.id] = draft_reply

    print(f"✅ Draft for ticket {ticket.id} generated and stored.")
    print("\n--- 🤖 AI DRAFT REPLY ---")
    print(draft_reply)
    print("-------------------------")
    print(f"To approve, call: GET /approve/{ticket.id}")

    # Immediately return a response to the user
    return {
        "status": "draft_generated",
        "ticket_id": ticket.id,
        "draft_reply": draft_reply,
        "approval_url": f"/approve/{ticket.id}",
        "reject_url": f"/reject/{ticket.id}" # We can add this for later
    }

# --- NEW: The /approve endpoint ---
@app.get("/approve/{ticket_id}")
async def approve_draft(ticket_id: int):
    # Check if a draft for this ticket exists
    if ticket_id not in PENDING_DRAFTS:
        return {"status": "error", "message": "No pending draft found for this ticket ID."}

    # Retrieve the draft
    approved_reply = PENDING_DRAFTS[ticket_id]
    
    print(f"\n✅ Draft for ticket {ticket_id} has been APPROVED by the user.")
    print("--- MESSAGE ---")
    print(approved_reply)
    print("---------------")

    # In the next step, we will add the code HERE to send this message to Slack.
    
    # Remove the draft from pending since it's now handled
    del PENDING_DRAFTS[ticket_id]
    
    return {"status": "draft_approved", "ticket_id": ticket_id, "sent_reply": approved_reply}