from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from models.database import get_db
from services.ingestion import CheckpointManager, DocumentSink, BatchResumableLoader, make_structured_docs
from services.incremental import IncrementalService
from connectors.csv_connector import load_csv
from connectors.sql_connector import fetch_sql
from connectors.api_connector import fetch_api


router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


class CSVIngestionRequest(BaseModel):
    file_path: str
    pipeline_id: str = Field(..., description="批处理ID用于断点续传")
    stable_fields: List[str] = Field(default_factory=lambda: ["id", "title", "content"])
    field_mapping: Optional[Dict[str, str]] = None


class BatchRunRequest(BaseModel):
    pipeline_id: str
    stable_fields: List[str]
    docs: List[Dict[str, Any]]


class SQLIngestionRequest(BaseModel):
    db_uri: str
    query: str
    pipeline_id: str
    stable_fields: List[str]
    field_mapping: Optional[Dict[str, str]] = None


class APIIngestionRequest(BaseModel):
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    pipeline_id: str
    stable_fields: List[str]
    field_mapping: Optional[Dict[str, str]] = None


class UpsertRequest(BaseModel):
    doc_id: int
    title: Optional[str] = None
    content: str
    metadata: Optional[str] = None


@router.get("/checkpoint/{pipeline_id}")
async def get_checkpoint(pipeline_id: str):
    ckpt = CheckpointManager().load(pipeline_id)
    return {"pipeline_id": pipeline_id, "checkpoint": ckpt and ckpt.__dict__}


@router.post("/csv")
async def ingest_csv(req: CSVIngestionRequest, db: Session = Depends(get_db)):
    rows = load_csv(req.file_path, field_mapping=req.field_mapping)
    docs = make_structured_docs(rows)
    sink = DocumentSink(db=db)
    loader = BatchResumableLoader(ckpt=CheckpointManager(), sink=sink, stable_fields=req.stable_fields)
    await loader.run(pipeline_id=req.pipeline_id, docs=docs, chunk_size=1000)
    return {"success": True, "count": len(docs)}


@router.post("/sql")
async def ingest_sql(req: SQLIngestionRequest, db: Session = Depends(get_db)):
    rows = fetch_sql(req.db_uri, req.query, field_mapping=req.field_mapping)
    docs = make_structured_docs(rows)
    sink = DocumentSink(db=db)
    loader = BatchResumableLoader(ckpt=CheckpointManager(), sink=sink, stable_fields=req.stable_fields)
    await loader.run(pipeline_id=req.pipeline_id, docs=docs, chunk_size=1000)
    return {"success": True, "count": len(docs)}


@router.post("/api")
async def ingest_api(req: APIIngestionRequest, db: Session = Depends(get_db)):
    rows = fetch_api(req.url, headers=req.headers, params=req.params, path=req.path, field_mapping=req.field_mapping)
    docs = make_structured_docs(rows)
    sink = DocumentSink(db=db)
    loader = BatchResumableLoader(ckpt=CheckpointManager(), sink=sink, stable_fields=req.stable_fields)
    await loader.run(pipeline_id=req.pipeline_id, docs=docs, chunk_size=1000)
    return {"success": True, "count": len(docs)}


@router.post("/run")
async def run_batch(req: BatchRunRequest, db: Session = Depends(get_db)):
    docs = make_structured_docs(req.docs)
    sink = DocumentSink(db=db)
    loader = BatchResumableLoader(ckpt=CheckpointManager(), sink=sink, stable_fields=req.stable_fields)
    await loader.run(pipeline_id=req.pipeline_id, docs=docs, chunk_size=1000)
    return {"success": True, "count": len(docs)}


@router.post("/incremental/upsert")
async def incremental_upsert(req: UpsertRequest, db: Session = Depends(get_db)):
    svc = IncrementalService(db=db)
    d = svc.upsert_document(doc_id=req.doc_id, title=req.title, content=req.content, metadata=req.metadata)
    return {"success": True, "document_id": d.id if d else None}


@router.post("/incremental/rebuild/{doc_id}")
async def incremental_rebuild(doc_id: int, db: Session = Depends(get_db)):
    svc = IncrementalService(db=db)
    d = svc.rebuild_document(doc_id)
    return {"success": True, "document_id": d.id if d else None}