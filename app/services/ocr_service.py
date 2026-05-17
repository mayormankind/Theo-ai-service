# src/ai-service/app/services/ocr_service.py
import pytesseract
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_bytes
from app.utils.text_preprocessing import clean_ocr_output

# Load environment variables
load_dotenv()

# Initialize OpenAI client 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

import re
from nltk.corpus import wordnet, stopwords

# Ensure we have common stop words loaded for basic word matching
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = set()

def evaluate_ocr_quality(text: str) -> bool:
    """
    Evaluates OCR output quality. 
    Returns True if quality is acceptable, False if it needs AI fallback.
    """
    # 1. Text length threshold
    if len(text.strip()) < 10:
        return False
        
    # 2. Ratio of readable words to noise
    # Calculate ratio of alphanumeric characters vs special/noise characters
    alphanumeric_chars = sum(c.isalnum() or c.isspace() for c in text)
    total_chars = len(text)
    
    if total_chars == 0:
        return False
        
    noise_ratio = 1.0 - (alphanumeric_chars / total_chars)
    if noise_ratio > 0.3:  # If more than 30% of text is noise
        return False
        
    # 3. Dictionary Validation (Lexical Check)
    # Tesseract often produces valid alphanumeric strings that are complete gibberish
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    if not words:
        return False
        
    valid_words_count = 0
    ACADEMIC_TERMS = {
        'plagiarism', 'academic', 'university', 'examination',
        'according', 'however', 'therefore', 'furthermore',
        'nigeria', 'federal', 'cope', 'ieee', 'apa', 'mla',
        'citation', 'reference', 'bibliography', 'hypothesis',
        'methodology', 'analysis', 'conclusion', 'introduction',
        'definition', 'explanation', 'describe', 'discuss',
        'evaluate', 'compare', 'contrast', 'explain', 'state',
        'outline', 'identify', 'distinguish', 'illustrate',
        'futa', 'unilag', 'uniben', 'abu', 'oau', 'eksu',
        'matric', 'coursework', 'semester', 'department',
        'lecturer', 'professor', 'student', 'faculty'
    }
    for word in words:
        # Check against stop words or WordNet dictionary
        if (word in STOP_WORDS or 
            word in ACADEMIC_TERMS or 
            wordnet.synsets(word)):
            valid_words_count += 1
            
    lexical_ratio = valid_words_count / len(words)
    
    # If less than 30% of extracted words are real English words, reject the output.
    # Lowered threshold for handwritten documents which often have lower OCR quality
    if lexical_ratio < 0.30:
        print(f"[OCR] Quality rejected. Lexical validity ratio too low: {lexical_ratio:.2f}")
        return False
        
    return True

def extract_text_with_gpt4(image_bytes: bytes) -> str:
    """
    Sends the image to GPT-4 Vision for text extraction.
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all readable handwritten text from this examination script. CRITICAL: Return ONLY the raw extracted text with ABSOLUTELY NO preamble, NO introductory sentences, NO commentary, NO explanations, and NO markdown formatting (no ``` fences). Start immediately with the first word from the document. Preserve the original structure and line breaks."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            # max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT-4 extraction failed: {e}")
        return ""

def process_file_to_images(file_bytes: bytes, filename: str) -> list:
    """
    Accepts uploaded script (image or PDF).
    Converts PDF pages into images if necessary.
    """
    if filename.lower().endswith('.pdf'):
        try:
            images = convert_from_bytes(file_bytes)
            image_bytes_list = []
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                image_bytes_list.append(img_byte_arr.getvalue())
            return image_bytes_list
        except Exception as e:
            print(f"Failed to process PDF: {e}")
            return []
    return [file_bytes]

def extract_text_hybrid(file_bytes: bytes, filename: str = "image.jpg") -> dict:
    """
    Main Hybrid OCR Pipeline:
    Step 1: Input handling
    Step 2: Primary OCR (Tesseract)
    Step 3: Quality Evaluation
    Step 4: AI Fallback (GPT-4)
    Step 5: Output Selection
    """
    image_bytes_list = process_file_to_images(file_bytes, filename)
    
    extracted_text_parts = []
    methods_used = []
    all_acceptable = True
    
    for img_bytes in image_bytes_list:
        try:
            # Step 2: Primary OCR
            image = Image.open(io.BytesIO(img_bytes))
            tesseract_text = pytesseract.image_to_string(image).strip()
            
            # Step 3: Evaluate OCR quality
            if evaluate_ocr_quality(tesseract_text):
                # Step 5: If Tesseract quality is acceptable -> use output
                extracted_text_parts.append(tesseract_text)
                methods_used.append("tesseract")
            else:
                # Step 4: AI Fallback
                all_acceptable = False
                gpt_text = extract_text_with_gpt4(img_bytes)
                
                # Clean GPT-4 output to remove conversational preambles
                if gpt_text:
                    gpt_text = clean_ocr_output(gpt_text)
                
                # Step 5: Replace with GPT-4 output
                # Fallback to tesseract text if GPT fails to return anything
                final_text = gpt_text if gpt_text else tesseract_text
                extracted_text_parts.append(final_text)
                methods_used.append("gpt4" if gpt_text else "tesseract (gpt4 failed)")
                
        except Exception as e:
            print(f"Error during extraction pipeline: {e}")
            
    full_text = "\n\n".join(extracted_text_parts)
    
    # Determine overall method
    if not methods_used:
        overall_method = "none"
    elif "gpt4" in methods_used and "tesseract" in methods_used:
        overall_method = "hybrid"
    else:
        overall_method = methods_used[0]
    
    return {
        "extracted_text": full_text,
        "extraction_method": overall_method,
        "confidence_flag": "acceptable" if all_acceptable else "low_quality_fallback_used"
    }