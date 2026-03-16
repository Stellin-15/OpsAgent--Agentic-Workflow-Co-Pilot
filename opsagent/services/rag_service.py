"""
RAG service — retrieval-augmented generation over runbook documents.

Phase 1: FAISS in-memory vector store, Google Gemini LLM.
Phase 4: This module is replaced by opsagent/ai/pipeline/rag_pipeline.py
         using pgvector, multi-model router, hybrid retrieval, and RAGAS eval.
"""

import os
from pathlib import Path

import structlog

from opsagent.config import Settings

log = structlog.get_logger(__name__)


class RagService:
    """
    Wraps the LangChain RAG chain.
    Initialised once at startup and stored in app.state.rag.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chain = None
        self._model_name = "models/gemini-2.0-flash"

    async def initialize(self) -> None:
        """
        Load runbooks, build FAISS index, and compile the RAG chain.
        Called from the FastAPI lifespan context.
        """
        runbooks_path = self._settings.runbooks_path
        log.info("rag_initializing", runbooks_path=runbooks_path)

        if not Path(runbooks_path).exists():
            log.warning("rag_runbooks_path_missing", path=runbooks_path)
            # Create an empty chain that returns a fallback message
            self._chain = None
            return

        # Lazy imports — keep startup fast if these aren't installed
        from langchain.prompts import ChatPromptTemplate
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_community.vectorstores import FAISS
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 1. Load runbook markdown files
        loader = DirectoryLoader(runbooks_path, glob="**/*.md")
        docs = loader.load()

        if not docs:
            log.warning("rag_no_documents_found", runbooks_path=runbooks_path)
            self._chain = None
            return

        log.info("rag_documents_loaded", count=len(docs))

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
        )
        splits = splitter.split_documents(docs)

        # 3. Embed and index
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self._settings.google_api_key,
        )
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self._settings.rag_retrieval_k}
        )

        # 4. LLM
        llm = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=0.3,
            google_api_key=self._settings.google_api_key,
        )

        # 5. Prompt — tuned for SRE incident response
        prompt = ChatPromptTemplate.from_template(
            """You are an expert Site Reliability Engineer (SRE) co-pilot.
An alert has fired and you must draft a clear, actionable response plan.

Use ONLY the runbook context below. If the context is not relevant, say you
could not find a matching runbook and recommend manual investigation.

Format your response as a numbered step-by-step action plan with:
- Each step on its own line
- Specific commands where applicable
- An estimated time for each step

RUNBOOK CONTEXT:
{context}

ALERT:
{alert_query}

RESPONSE PLAN:
"""
        )

        # 6. Assemble chain
        self._chain = (
            {"context": retriever, "alert_query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        log.info("rag_initialized", model=self._model_name, chunks=len(splits))

    def is_ready(self) -> bool:
        return self._chain is not None

    async def generate_draft(
        self,
        alert_name: str,
        labels: dict,
        description: str,
    ) -> str:
        """
        Generate a response plan draft for the given alert.
        Returns the draft text.
        """
        if not self._chain:
            return (
                "⚠️ RAG chain not initialised. No runbooks found or GOOGLE_API_KEY missing. "
                "Please add runbooks to the /runbooks directory and restart."
            )

        query = (
            f"Alert: {alert_name}\n"
            f"Labels: {labels}\n"
            f"Description: {description or 'No description provided.'}"
        )

        log.info("rag_generating_draft", alert_name=alert_name)
        draft = self._chain.invoke(query)
        log.info("rag_draft_generated", alert_name=alert_name, length=len(draft))
        return draft
