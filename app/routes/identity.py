from fastapi import APIRouter
from pydantic import BaseModel
import httpx
import re
from app.services.ocr_service import (
  extract_page_text, 
  process_file_to_images
)

router = APIRouter()

class IdentityRequest(BaseModel):
    url: str

def extract_matric(text: str) -> str | None:
    normalised = re.sub(r'\s*/\s*', '/', text.upper())
    pattern = r'\b[A-Z]{2,5}/\d{2}/\d{3,5}\b'
    match = re.search(pattern, normalised)
    return match.group(0) if match else None

@router.post("/extract-identity")
async def extract_identity(payload: IdentityRequest):
    try:
        # Download file
        async with httpx.AsyncClient(timeout=30.0, verify=False) as c:
            resp = await c.get(payload.url)
            resp.raise_for_status()
            file_bytes = resp.content

        # Strip query string from url
        url_no_query = payload.url.split('?')[0]
        filename = url_no_query.split('/')[-1] or 'file'
        
        # Get first page only
        pages = process_file_to_images(
          file_bytes, filename
        )
        if not pages:
            return { "matric": None, "confidence": 0 }

        # OCR first page only
        page_text = extract_page_text(pages[0])
        matric = extract_matric(page_text)

        return {
            "matric": matric,
            "confidence": 85 if matric else 0,
            "raw_text": page_text[:300]
        }
    except Exception as e:
        print(f"[Identity] extraction failed: {e}")
        return { "matric": None, "confidence": 0 }
