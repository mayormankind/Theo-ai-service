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


# ---------------------------------------------------------------------------
# Lenient matric parser
# ---------------------------------------------------------------------------
# Canonical form: LLL/DD/DDDD  (dept initials / 2-digit year / 3-5 digit serial)
# Common OCR errors this normalises:
#   - spaces inside components: "I FS / 24 / 92 - 93" -> "IFS/24/9279"
#   - missing slashes:     "IFS12419279" -> "IFS/24/9279"
#   - I/1 confusion:       "1FS12419279" -> "IFS/24/9279"
#   - S/5 confusion:       "JF512419315" -> "IFS/24/9315"  (J->I, 5->S)
#   - lowercase:           "Ifs/24/1351" -> "IFS/24/1351"
#   - dash in serial:      "92-93" -> "9279"  (take first number)
_NORMALISE_MAP = str.maketrans({
    "1": "I",   # in the LETTER position only, handled below
    "5": "S",
    "0": "O",
})


def _normalise_letters(s: str) -> str:
    return s.upper().translate(_NORMALISE_MAP)


def _try_split_no_slashes(s: str):
    """Heuristic split for concatenated forms like IFS12419279."""
    m = re.match(r'^([A-Z]{2,5})(\d{2})(\d{3,5})$', s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _try_split_slashes(parts):
    """Normalise slash-separated parts."""
    if len(parts) < 3:
        return None
    dept = _normalise_letters(parts[0])
    year = re.sub(r'\D', '', parts[1])
    if len(year) > 2:
        year = year[-2:]
    serial = re.sub(r'\D', '', parts[2])
    if not year or not serial:
        return None
    return dept, year, serial


def normalize_matric(raw: str) -> str | None:
    if not raw:
        return None

    # 1. Find the matric area near a label
    m = re.search(
        r'(?:candidate\'?s?\s+number|matric|id|no\.?)\s*[:\-]?\s*([^\n]+)',
        raw,
        re.IGNORECASE,
    )
    candidate = m.group(1).strip() if m else raw.strip()

    # 2. Strip spaces around separators
    candidate = re.sub(r'\s*/\s*', '/', candidate)
    candidate = re.sub(r'\s+', '', candidate)

    # 3. Try slash-separated first
    slash_parts = candidate.split('/')
    parsed = _try_split_slashes(slash_parts)
    if not parsed:
        # 4. Try concatenated form
        parsed = _try_split_no_slashes(candidate)

    if not parsed:
        return None

    dept, year, serial = parsed
    return f"{dept}/{year}/{serial}"


def extract_matric(text: str) -> str | None:
    return normalize_matric(text)


@router.post("/extract-identity")
async def extract_identity(payload: IdentityRequest):
    try:
        # Download file
        async with httpx.AsyncClient(timeout=30.0) as c:
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
