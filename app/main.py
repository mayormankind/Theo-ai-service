# src/ai-service/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import nltk

# Load environment variables
load_dotenv()

# Download necessary NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

# Get port from environment or default to 8000
port = int(os.getenv("PORT", 8000))
from app.routes import ocr, similarity, grading, rubric_extraction
from app.services.segmentation_service import segment_answers
from app.models.request_models import SegmentRequest

app = FastAPI(
    title="Intelligent Assessment System API",
    description="Automated Grading of Theoretical Examination Scripts in Nigerian Universities using Sentence-BERT (OpenAI fallback).",
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
