from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ocr_service import extract_text_hybrid

router = APIRouter()

@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image or PDF file upload and returns the extracted raw text using a Hybrid OCR Pipeline (Tesseract with GPT-4 Vision fallback).
    """
    try:
        contents = await file.read()
        filename = file.filename or "image.jpg"
        result = extract_text_hybrid(contents, filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
