# src/ai-service/app/services/segmentation_service.py
import re
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))

def segment_answers(raw_text: str) -> dict:
    """
    Segments the raw OCR text into individual questions using regular expressions.
    Detects markers like: "Q1", "Question 1", "1(a)", "1a" when at the start of a line.

    The pattern is anchored to line-start (or explicit "Question"/"Q" prefix) to
    avoid matching numbered lists, years, or decimal numbers inside answer text.
    """
    pattern = re.compile(
        r'(?im)'                                  # case-insensitive, multiline
        r'(?:'
        r'(?:^|\n)\s*question\s*\d+[a-z]?'       # "Question 1" / "Question 1a" at line start
        r'|(?:^|\n)\s*q\d+[a-z]?(?!\w)'          # "Q1" / "Q2b" at line start (not inside a word)
        r'|(?:^|\n)\s*\d+\s*\([a-z]\)'           # "1(a)", "2(b)" at line start
        r'|(?:^|\n)\s*\d+[a-z]\b'                # "1a", "2b" at line start
        r')'
    )
    matches = list(pattern.finditer(raw_text))
    
    # Intelligent LLM Fallback if Regex fails
    if not matches and len(raw_text.strip()) > 20:
        return intelligent_segmentation_fallback(raw_text)
    elif not matches:
        return {"Uncategorized": raw_text.strip()}
    
    segments = {}
    for i in range(len(matches)):
        start_idx = matches[i].end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(raw_text)
        
        q_label = matches[i].group().strip()
        answer_text = raw_text[start_idx:end_idx].strip()
        
        if answer_text:
            segments[q_label] = answer_text
            
    return segments

def intelligent_segmentation_fallback(raw_text: str) -> dict:
    """
    Uses gpt-4o-mini to intelligently segment text if standard regex structures fail.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that segments exam script texts. Return ONLY a valid JSON object where keys are the question numbers (e.g. 'Q1', '2(a)') and values are the corresponding answer texts."
                },
                {"role": "user", "content": raw_text}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Intelligent segmentation fallback failed: {e}")
        return {"Uncategorized": raw_text.strip()}
