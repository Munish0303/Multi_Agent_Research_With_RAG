"""
Optional REST API server for the Multi-Agent Research System.
Run with: uvicorn api:app --reload
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from workflow import run_research
from rag.pipeline import RAGPipeline
from config.settings import UPLOAD_DIR

app = FastAPI(
    title="Multi-Agent Research System",
    description="Autonomous research using collaborative AI agents",
    version="1.0.0",
)

executor = ThreadPoolExecutor(max_workers=2)
jobs: dict = {}
rag = RAGPipeline()


class ResearchRequest(BaseModel):
    topic: str
    use_rag: bool = True


class ResearchResponse(BaseModel):
    job_id: str
    status: str
    message: str


@app.get("/")
def root():
    return {"name": "Multi-Agent Research System", "status": "running", "docs": "/docs"}


@app.get("/status")
def system_status():
    return {
        "documents_indexed": rag.collection_count(),
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"]),
    }


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "topic": request.topic, "result": None, "error": None}

    def run_job():
        try:
            result = run_research(request.topic, rag=rag if request.use_rag else None)
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "complete"
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "failed"

    background_tasks.add_task(run_job)
    return ResearchResponse(job_id=job_id, status="running", message=f"Research started on: {request.topic}")


@app.get("/research/{job_id}")
def get_research_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, MD files supported")

    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        count = rag.ingest_file(path)
        return {"filename": file.filename, "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
def list_jobs():
    return [{"job_id": k, "topic": v["topic"], "status": v["status"]} for k, v in jobs.items()]
