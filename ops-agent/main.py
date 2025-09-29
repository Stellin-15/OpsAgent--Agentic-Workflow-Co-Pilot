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

rag_chain = None

class Ticket(BaseModel):
    id: int
    title: str
    description: str

app = FastAPI(title="OpsAgent - Powered by Gemini")

@app.on_event("startup")
def startup_event():
    print("🚀 Server is starting up. Beginning knowledge base ingestion...")
    global rag_chain
    
    sops_folder_path = "/app/sops" 

    loader = DirectoryLoader(sops_folder_path, glob="**/*.md")
    docs = loader.load()
    print(f"📚 Found and loaded {len(docs)} SOP document(s).")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"📄 Split documents into {len(splits)} chunks.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print("🧠 Building the FAISS vector store with Google embeddings...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("✅ Knowledge base built.")

    print("🔗 Building the RAG chain...")

    # THIS IS THE FINAL CORRECTED LINE
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
    query = f"Title: {ticket.title}\nDescription: {ticket.description}"
    
    print(f"\n🔎 Received query for ticket {ticket.id}: {query}")
    print("🤔 Thinking...")

    draft_reply = rag_chain.invoke(query)

    print("\n--- 🤖 AI DRAFT REPLY ---")
    print(draft_reply)
    print("-------------------------\n")

    return {
        "status": "draft generated", 
        "ticket_id": ticket.id,
        "draft_reply": draft_reply
    }