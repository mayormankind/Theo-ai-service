# src/ai-service/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

# Get port from environment or default to 8000
port = int(os.getenv("PORT", 8000))
from app.routes import ocr, similarity, grading, rubric_extraction
from app.services.segmentation_service import segment_answers
from app.models.request_models import SegmentRequest
from app.routes.identity import router as identity_router

app = FastAPI(
    title="Intelligent Assessment System API",
    description="Automated Grading of Theoretical Examination Scripts in Nigerian Universities. Uses GPT-4o-mini Vision for handwritten text extraction and OpenAI text-embedding-3-small for semantic similarity scoring.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach Modular Routers
app.include_router(ocr.router, tags=["OCR"])
app.include_router(similarity.router, tags=["Similarity"])
app.include_router(grading.router, tags=["Grading Pipeline"])
app.include_router(rubric_extraction.router, tags=["Rubric Extraction"])
app.include_router(identity_router, tags=["Identity"])

@app.get("/", tags=["Health"])
def health_check():
    """
    Root health check endpoint.
    """
    return {"status": "ok", "message": "TheoGrader AI Service API is running."}

@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
def health():
    return {"ok": True}
    
@app.post("/segment", tags=["Segmentation"])
def segment_endpoint(req: SegmentRequest):
    """
    Receives raw text and segments it into structured answers based on regex patterns.
    """
    segments = segment_answers(req.raw_text)
    return segments
