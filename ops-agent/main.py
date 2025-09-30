import os
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slack_sdk import WebClient
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
slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL")

class Ticket(BaseModel):
    id: int
    title: str
    description: str

# --- 2. FastAPI App & Router Initialization ---
app = FastAPI(title="OpsAgent")
api_router = APIRouter(prefix="/api")


# --- 3. Startup Logic ---
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
    llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # --- THIS IS THE FULL, CORRECT PROMPT ---
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


# --- 4. API Endpoints (on the router) ---
@api_router.post("/tickets")
async def process_ticket(ticket: Ticket):
    query = f"Title: {ticket.title}\nDescription: {ticket.description}"
    print(f"\n🔎 Received query for ticket {ticket.id}: {query}")
    draft_reply = rag_chain.invoke(query)
    PENDING_DRAFTS[ticket.id] = draft_reply
    print(f"✅ Draft for ticket {ticket.id} generated.")
    return {
        "status": "draft_generated",
        "ticket_id": ticket.id,
        "draft_reply": draft_reply,
    }

@api_router.get("/approve/{ticket_id}")
async def approve_draft(ticket_id: int):
    if ticket_id not in PENDING_DRAFTS:
        return {"status": "error", "message": "No pending draft found for this ticket ID."}
    
    # Use .pop() to retrieve the value AND remove it from the dictionary in one step
    approved_reply = PENDING_DRAFTS.pop(ticket_id)
    print(f"\n✅ Draft for ticket {ticket_id} has been APPROVED.")

    try:
        print(f"...\nPosting message to Slack channel: {SLACK_CHANNEL_ID}")
        slack_client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=approved_reply)
        print("✅ Message posted successfully to Slack!")
    except SlackApiError as e:
        print(f"❌ Error posting to Slack: {e.response['error']}")
        # If sending fails, put the draft back so the user can try approving again
        PENDING_DRAFTS[ticket_id] = approved_reply 
        return {"status": "error", "message": f"Failed to post to Slack: {e.response['error']}"}

    return {"status": "draft_approved_and_sent", "ticket_id": ticket_id}

# --- 5. Final App Setup ---
# Include all the API routes from our router into the main application
app.include_router(api_router)

# Mount the static files directory to serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")