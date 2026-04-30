from fastapi import FastAPI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
from app.routes import ocr, similarity, grading
from app.services.segmentation_service import segment_answers
from app.models.request_models import SegmentRequest

app = FastAPI(
    title="Intelligent Assessment System API",
    description="Automated Grading of Theoretical Examination Scripts in Nigerian Universities using Sentence-BERT.",
    version="1.0.0"
)

# Attach Modular Routers
app.include_router(ocr.router, tags=["OCR"])
app.include_router(similarity.router, tags=["Similarity"])
app.include_router(grading.router, tags=["Grading Pipeline"])

@app.get("/", tags=["Health"])
def health_check():
    """
    Root health check endpoint.
    """
    return {"status": "ok", "message": "Intelligent Assessment System API is running."}

@app.post("/segment", tags=["Segmentation"])
def segment_endpoint(req: SegmentRequest):
    """
    Receives raw text and segments it into structured answers based on regex patterns.
    """
    segments = segment_answers(req.raw_text)
    return segments
