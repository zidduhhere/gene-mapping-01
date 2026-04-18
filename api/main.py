"""PharmaAI Predictor — FastAPI backend."""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import auth, predict, explain, report, drugs
from api.services.model_service import model_service

app = FastAPI(title="PharmaAI Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(drugs.router, prefix="/drugs", tags=["drugs"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(explain.router, prefix="/explain", tags=["explain"])
app.include_router(report.router, prefix="/report", tags=["report"])


@app.on_event("startup")
async def startup():
    model_service.load()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_service.is_loaded()}
