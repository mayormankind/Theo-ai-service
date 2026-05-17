# src/ai-service/app/routes/rubric_extraction.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import json
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

router = APIRouter()

class ExtractedQuestion(BaseModel):
    questionNumber: str
    questionText: str
    maxScore: int
    parts: List[dict]

class ExtractedRubric(BaseModel):
    title: str
    description: Optional[str] = None
    courseCode: Optional[str] = None
    examType: Optional[str] = None
    questions: List[ExtractedQuestion]
    totalMarks: int

class ExtractionResult(BaseModel):
    success: bool
    rubric: Optional[ExtractedRubric] = None
    error: Optional[str] = None
    confidence: Optional[float] = None

async def extract_from_text_internal(text: str, filename: str = "document"):
    """
    Internal function to extract rubric from text content.
    """
    prompt = f"""
Extract the complete rubric structure from this marking scheme text and return as structured JSON with the following format:

{{
  "title": "Exam Title",
  "description": "Brief description",
  "courseCode": "Course Code (if available)",
  "examType": "Exam Type (if available)",
  "questions": [
    {{
      "questionNumber": "Q1",
      "questionText": "Full question text",
      "maxScore": 20,
      "parts": [
        {{
          "label": "a",
          "expectedAnswer": "Expected answer for this part",
          "keyPoints": ["key concept 1", "key concept 2"],
          "marks": 10
        }}
      ]
    }}
  ],
  "totalMarks": 100
}}

Text to analyze:
{text[:10000]}  # Limit to first 10k chars to avoid token limits

Guidelines:
- Extract ALL questions and sub-parts thoroughly
- Identify marks allocation for each part explicitly
- Extract key concepts/points that should be mentioned in answers
- If marks aren't explicitly stated, estimate based on question complexity
- Include both complete question text and expected answers
- Be thorough but don't invent information that's not clearly present
- Look for question numbers like Q1, 1(a), Question 1, etc.
- Identify marking schemes, answer keys, or model answers
- Extract any weight distributions or mark allocations
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()
    
    # Log the raw response for debugging
    print(f"OpenAI Response (first 500 chars): {content[:500]}")
    print(f"Full response length: {len(content)}")
    
    # Parse JSON from response
    json_match = content.find('{')
    if json_match == -1:
        raise ValueError(f"No JSON found in response. Raw content: {content[:1000]}")
    
    json_content = content[json_match:]
    try:
        rubric_data = json.loads(json_content)
    except json.JSONDecodeError:
        import re
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, content)
        if json_match:
            rubric_data = json.loads(json_match.group())
        else:
            raise ValueError("Could not parse JSON from response")

    # Validate and structure the response
    extracted_rubric = ExtractedRubric(**rubric_data)
    
    # Ensure all parts have labels
    for q in extracted_rubric.questions:
        for idx, part in enumerate(q.parts):
            if not part.get("label"):
                part["label"] = chr(97 + idx)  # a, b, c, etc.
    
    return ExtractionResult(
        success=True,
        rubric=extracted_rubric,
        confidence=0.80
    )

@router.post("/extract/from-document", response_model=ExtractionResult)
async def extract_from_document(file: UploadFile = File(...)):
    """
    Extract rubric structure from uploaded document (PDF, Word, or image).
    Uses OpenAI Vision API for images and text extraction for PDFs.
    """
    try:
        # Read file content
        file_bytes = await file.read()
        filename = file.filename or "document"
        
        # Determine file type
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ""
        is_image = file_extension in ('jpg', 'jpeg', 'png', 'bmp', 'gif')
        is_pdf = file_extension == 'pdf'
        
        # Extract text content
        if is_pdf:
            # Extract text from PDF
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            if not text_content.strip():
                return ExtractionResult(
                    success=False,
                    error="Could not extract text from PDF. The PDF might be image-based or corrupted."
                )
            
            # Use text extraction endpoint for PDFs
            return await extract_from_text_internal(text_content, filename)
        
        elif is_image:
            # Convert image to base64 for vision API
            base64_content = base64.b64encode(file_bytes).decode('utf-8')
            
            prompt = """
Extract the complete rubric structure from this marking scheme document and return as structured JSON with the following format:

{
  "title": "Exam Title",
  "description": "Brief description",
  "courseCode": "Course Code (if available)",
  "examType": "Exam Type (if available)",
  "questions": [
    {
      "questionNumber": "Q1",
      "questionText": "Full question text",
      "maxScore": 20,
      "parts": [
        {
          "label": "a",
          "expectedAnswer": "Expected answer for this part",
          "keyPoints": ["key concept 1", "key concept 2"],
          "marks": 10
        }
      ]
    }
  ],
  "totalMarks": 100
}

Guidelines:
- Extract ALL questions and sub-parts thoroughly
- Identify marks allocation for each part explicitly
- Extract key concepts/points that should be mentioned in answers
- If marks aren't explicitly stated, estimate based on question complexity
- Include both complete question text and expected answers
- Be thorough but don't invent information that's not clearly present
- Look for question numbers like Q1, 1(a), Question 1, etc.
- Identify marking schemes, answer keys, or model answers
- Extract any weight distributions or mark allocations
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:{file.content_type};base64,{base64_content}"
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            print(f"OpenAI Response (first 500 chars): {content[:500]}")
            print(f"Full response length: {len(content)}")
            
            # Parse JSON from response
            json_match = content.find('{')
            if json_match == -1:
                raise ValueError(f"No JSON found in response. Raw content: {content[:1000]}")
            
            json_content = content[json_match:]
            try:
                rubric_data = json.loads(json_content)
            except json.JSONDecodeError:
                import re
                json_pattern = r'\{[\s\S]*\}'
                json_match = re.search(json_pattern, content)
                if json_match:
                    rubric_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from response")

            extracted_rubric = ExtractedRubric(**rubric_data)
            
            # Ensure all parts have labels
            for q in extracted_rubric.questions:
                for idx, part in enumerate(q.parts):
                    if not part.get("label"):
                        part["label"] = chr(97 + idx)  # a, b, c, etc.
            
            return ExtractionResult(
                success=True,
                rubric=extracted_rubric,
                confidence=0.85
            )
        
        else:
            return ExtractionResult(
                success=False,
                error=f"Unsupported file format: {file_extension}. Please upload PDF, JPG, JPEG, or PNG files."
            )

    except Exception as e:
        print(f"Error extracting from document: {e}")
        import traceback
        traceback.print_exc()
        
        return ExtractionResult(
            success=False,
            error=f"Failed to extract rubric: {str(e)}"
        )

@router.post("/extract/from-text", response_model=ExtractionResult)
async def extract_from_text(text: str = Form(...)):
    """
    Extract rubric structure from pasted text.
    Uses OpenAI to parse the text and extract structured rubric data.
    """
    try:
        if not text.strip():
            return ExtractionResult(
                success=False,
                error="No text provided for extraction"
            )

        prompt = f"""
Extract the complete rubric structure from this marking scheme text and return as structured JSON with the following format:

{{
  "title": "Exam Title",
  "description": "Brief description",
  "courseCode": "Course Code (if available)",
  "examType": "Exam Type (if available)",
  "questions": [
    {{
      "questionNumber": "Q1",
      "questionText": "Full question text",
      "maxScore": 20,
      "parts": [
        {{
          "label": "a",
          "expectedAnswer": "Expected answer for this part",
          "keyPoints": ["key concept 1", "key concept 2"],
          "marks": 10
        }}
      ]
    }}
  ],
  "totalMarks": 100
}}

Text to analyze:
{text}

Guidelines:
- Extract ALL questions and sub-parts thoroughly
- Identify marks allocation for each part explicitly
- Extract key concepts/points that should be mentioned in answers
- If marks aren't explicitly stated, estimate based on question complexity
- Include both complete question text and expected answers
- Be thorough but don't invent information that's not clearly present
- Look for question numbers like Q1, 1(a), Question 1, etc.
- Identify marking schemes, answer keys, or model answers
- Extract any weight distributions or mark allocations
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        content = response.choices[0].message.content.strip()
        
        # Parse JSON from response
        json_match = content.find('{')
        if json_match == -1:
            raise ValueError("No JSON found in response")
        
        json_content = content[json_match:]
        try:
            rubric_data = json.loads(json_content)
        except json.JSONDecodeError:
            # Try to extract JSON more carefully
            import re
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, content)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from response")

        # Validate and structure the response
        extracted_rubric = ExtractedRubric(**rubric_data)
        
        # Ensure all parts have labels
        for q in extracted_rubric.questions:
            for idx, part in enumerate(q.parts):
                if not part.get("label"):
                    part["label"] = chr(97 + idx)  # a, b, c, etc.
        
        return ExtractionResult(
            success=True,
            rubric=extracted_rubric,
            confidence=0.80
        )

    except Exception as e:
        print(f"Error extracting from text: {e}")
        return ExtractionResult(
            success=False,
            error=str(e)
        )

@router.get("/health")
def health_check():
    """
    Health check endpoint for rubric extraction service.
    """
    return {"status": "ok", "service": "rubric-extraction"}
