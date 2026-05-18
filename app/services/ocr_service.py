# src/ai-service/app/services/ocr_service.py
import base64
import io
import os
from openai import OpenAI
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

TRANSCRIPTION_PROMPT = """You are a faithful transcription \
assistant for handwritten university examination scripts.

Your task: Transcribe EXACTLY what is handwritten on this \
examination script page.

CRITICAL RULES:
1. Copy the text VERBATIM as written — do not paraphrase, \
   summarise, correct, or improve the student's writing
2. If a word is unclear, make your best attempt to read it \
   and transcribe it exactly as it appears — do not \
   substitute a different word that seems to make more sense
3. If text is completely illegible, write [illegible] in \
   that position — do not guess
4. Preserve structure: maintain line breaks, question \
   numbers, and the order content appears on the page
5. Do NOT add any commentary, explanations, or preamble
6. Do NOT fix spelling or grammar errors — transcribe \
   exactly what the student wrote, errors included
7. Include ALL visible text: name, matric number, course \
   code, date, question numbers, and all answers

Return ONLY the transcribed text. Nothing else."""


def extract_student_id(text: str) -> str:
    """
    Extracts Nigerian university matric number from text.
    Handles formats: IFS/20/4986, CSC/21/1234, etc.
    """
    pattern = re.compile(
        r'(?i)(?:matric|id|no\.?)?\s*[:\-]*\s*'
        r'((?:[a-z]{2,5}/)?[0-9]{2,4}/[a-z]{2,5}/[0-9]{3,4}'
        r'|[a-z]{2,5}/[0-9]{2}/[0-9]{3,4}'
        r'|[0-9]{2}/[a-z0-9]+)'
    )
    match = pattern.search(text)
    if match:
        return match.group(1).upper()
    return "UNKNOWN"


def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def extract_page_text(image_bytes: bytes) -> str:
    """
    Extract text from a single image using 
    GPT-4o-mini Vision.
    Returns the transcribed text string.
    """
    base64_image = image_bytes_to_base64(image_bytes)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": TRANSCRIPTION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[OCR] GPT-4o-mini extraction failed: {e}")
        return ""


def process_file_to_images(
    file_bytes: bytes, 
    filename: str
) -> list:
    """
    Convert uploaded file to list of image byte arrays.
    PDFs are converted page by page.
    Images are returned as-is in a single-item list.
    """
    if filename.lower().endswith('.pdf'):
        try:
            images = convert_from_bytes(
                file_bytes, 
                dpi=200
            )
            image_bytes_list = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=95)
                image_bytes_list.append(buf.getvalue())
            return image_bytes_list
        except Exception as e:
            print(f"[OCR] PDF conversion failed: {e}")
            return []
    else:
        return [file_bytes]


def extract_text_hybrid(
    file_bytes: bytes, 
    filename: str = "image.jpg"
) -> dict:
    """
    Main OCR pipeline using GPT-4o-mini Vision.
    Processes each page and combines results.

    Returns:
        extracted_text: full transcribed text
        extraction_method: always "gpt4-mini"
        confidence_flag: "acceptable" or 
                         "low_quality_fallback_used"
    """
    image_bytes_list = process_file_to_images(
        file_bytes, filename
    )

    if not image_bytes_list:
        return {
            "extracted_text": "",
            "extraction_method": "none",
            "confidence_flag": "low_quality_fallback_used"
        }

    extracted_pages = []

    for i, img_bytes in enumerate(image_bytes_list):
        print(f"[OCR] Processing page {i + 1} of "
              f"{len(image_bytes_list)}...")
        page_text = extract_page_text(img_bytes)

        if page_text:
            extracted_pages.append(page_text)
        else:
            print(f"[OCR] Page {i + 1} returned no text")
            # Insert a placeholder so the lecturer knows a page failed rather than silently losing it
            extracted_pages.append(
                f"[PAGE {i + 1} EXTRACTION FAILED - "
                f"please review original script]"
            )

    full_text = "\n\n".join(extracted_pages)
    
    any_page_failed = any(
        "EXTRACTION FAILED" in p 
        for p in extracted_pages
    )

    confidence_flag = (
        "low_quality_fallback_used"
        if any_page_failed or not full_text.strip()
        else "acceptable"
    )

    return {
        "extracted_text": full_text,
        "extraction_method": "gpt4-mini",
        "confidence_flag": confidence_flag
    }