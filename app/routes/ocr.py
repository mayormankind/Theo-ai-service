from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
from app.services.ocr_service import extract_text_hybrid

router = APIRouter()

class OcrFromUrlRequest(BaseModel):
    url: str

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

@router.post("/ocr-from-url")
async def ocr_from_url_endpoint(payload: OcrFromUrlRequest):
    """
    Accepts a file URL, downloads the file, and returns the extracted text using the Hybrid OCR Pipeline.
    """
    try:
        file_url = payload.url
        
        # Download file from URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            contents = response.content
        
        # Extract filename from URL or use default
        filename = file_url.split("/")[-1] or "downloaded_file"
        
        # Run OCR pipeline on downloaded bytes
        result = extract_text_hybrid(contents, filename)
        return result
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
