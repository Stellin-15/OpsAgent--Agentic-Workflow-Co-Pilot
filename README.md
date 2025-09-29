# OpsAgent — Agentic Workflow Co-Pilot

## Overview
**OpsAgent** turns raw IT tickets into helpful draft replies using your SOPs/runbooks.  

**MVP flow:**  
`Ticket → Retrieve SOP context (FAISS) → LLM draft → Human approval → (Optional) Slack post`

---

## Features (MVP)
- **/tickets (POST):** Accept `{id, title, description}` and return a draft reply.  
- **SOP ingestion:** Load `.md` / `.txt` files from `sops/`, chunk, embed, store in FAISS.  
- **Retriever:** Top-k hybrid-ish (start with embeddings; add keyword later).  
- **Drafting agent:** LLM (**GPT-4o-mini** or **LLaMA-3**) crafts Slack-ready answers grounded in SOP chunks.  
- **Reviewer step:** CLI approve/reject (or mock with `POST /approve`).  
- **Dockerized service** with `.env` config.  
- (Optional) **Slack integration** to post approved messages.  

---

## Tech Stack
- **Backend:** Python, Flask (API)  
- **AI/ML:** LangChain, FAISS (`faiss-cpu`), OpenAI (or local LLaMA-3)  
- **Config/Utils:** `python-dotenv`, `pydantic`  
- **Optional:** `slack_sdk` for Slack posting  

---

## Project Structure
```

opsagent/
app.py
ops/
ingest.py
retriever.py
draft.py
sops/                # your SOP .md/.txt files
store/faiss_index/   # persisted FAISS
.env
requirements.txt
Dockerfile

````

---

## Environment Variables
Create a `.env` file in the project root with the following values:
```bash
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
TOP_K=4
FAISS_DIR=store/faiss_index
````

---

## Install & Run (Local)

```bash
# Create and activate virtual environment
python -m venv venv && source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## Docker

```bash
# Build the Docker image
docker build -t opsagent .

# Run the container
docker run --env-file .env -p 8000:8000 opsagent
```

```

---

✅ This is complete and **ready to paste** into your repo.  

Do you want me to also add a **Usage Examples** section (like sample `curl` requests to `/tickets`) so anyone can test it right after setup?
```

