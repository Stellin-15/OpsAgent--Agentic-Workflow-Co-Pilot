from fastapi import FastAPI

app = FastAPI(title="OpsAgent")

@app.get("/")
def read_root():
    return {"service": "OpsAgent", "status": "Hello World! Setup is working!"}