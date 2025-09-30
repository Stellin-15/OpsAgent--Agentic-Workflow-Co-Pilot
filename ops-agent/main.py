import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from slack_sdk import WebClient # <-- NEW: Import the Slack client
from slack_sdk.errors import SlackApiError

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
PENDING_DRAFTS = {}

# --- NEW: Initialize the Slack Client ---
# It will automatically read the SLACK_BOT_TOKEN from our .env file
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL")


class Ticket(BaseModel):
    id: int
    title: str
    description: str

app = FastAPI(title="OpsAgent - Powered by Gemini")

@app.on_event("startup")
def startup_event():
    # (No changes to the startup logic)
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


@app.get("/")
def read_root():
    return {"service": "OpsAgent", "status": "running", "model_provider": "Google Gemini"}

@app.post("/tickets")
async def process_ticket(ticket: Ticket):
    # (No changes to the ticket processing logic)
    query = f"Title: {ticket.title}\nDescription: {ticket.description}"
    print(f"\n🔎 Received query for ticket {ticket.id}: {query}")
    print("🤔 Thinking...")
    draft_reply = rag_chain.invoke(query)
    PENDING_DRAFTS[ticket.id] = draft_reply
    print(f"✅ Draft for ticket {ticket.id} generated and stored.")
    print("\n--- 🤖 AI DRAFT REPLY ---")
    print(draft_reply)
    print("-------------------------")
    print(f"To approve, call: GET /approve/{ticket.id}")
    return {
        "status": "draft_generated",
        "ticket_id": ticket.id,
        "draft_reply": draft_reply,
        "approval_url": f"/approve/{ticket.id}",
        "reject_url": f"/reject/{ticket.id}"
    }

# --- MODIFIED: The /approve endpoint now posts to Slack ---
@app.get("/approve/{ticket_id}")
async def approve_draft(ticket_id: int):
    if ticket_id not in PENDING_DRAFTS:
        return {"status": "error", "message": "No pending draft found for this ticket ID."}

    approved_reply = PENDING_DRAFTS[ticket_id]
    print(f"\n✅ Draft for ticket {ticket_id} has been APPROVED by the user.")
    
    # --- NEW: Post the message to Slack ---
    try:
        print(f"...\nPosting message to Slack channel: {SLACK_CHANNEL_ID}")
        response = slack_client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=approved_reply
        )
        print("✅ Message posted successfully to Slack!")
    except SlackApiError as e:
        print(f"❌ Error posting to Slack: {e.response['error']}")
        return {"status": "error", "message": f"Failed to post to Slack: {e.response['error']}"}

    del PENDING_DRAFTS[ticket_id]
    
    return {"status": "draft_approved_and_sent", "ticket_id": ticket_id}