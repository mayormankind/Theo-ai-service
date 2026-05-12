from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import json
import os
from fastapi.concurrency import run_in_threadpool

from app.models.response_models import GradeResponse, QuestionResult
from app.services.ocr_service import extract_text_hybrid
from app.services.segmentation_service import segment_answers
from app.services.embedding_service import get_embeddings
from app.services.scoring_service import calculate_similarity, calculate_final_score
from app.utils.text_preprocessing import preprocess_text, extract_student_id

router = APIRouter()


@router.post("/grade", response_model=GradeResponse)
async def grade_endpoint(
    file: UploadFile = File(...),
    rubric_str: str = Form(..., description="JSON string of rubric points with weights")
):
    """
    Single script grading pipeline.
    """
    try:
        rubric_data = json.loads(rubric_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid rubric JSON format")

    # Step 1: OCR - Run in threadpool to free event loop!
    image_bytes = await file.read()
    filename = file.filename or "image.jpg"
    ocr_result = await run_in_threadpool(extract_text_hybrid, image_bytes, filename)
    raw_text = ocr_result.get("extracted_text", "")
    
    if not raw_text:
        return GradeResponse(student_id="UNKNOWN", questions=[])
        
    student_id = extract_student_id(raw_text)
    
    # Step 2: Segmentation
    segmented_answers = segment_answers(raw_text)
    
    results = []
    
    for question, answer_text in segmented_answers.items():
        if not answer_text.strip():
            results.append(QuestionResult(question=question, score=0.0, confidence=0.0, breakdown=[]))
            continue
            
        clean_student_answer = preprocess_text(answer_text)
        
        # Normalization for matching (e.g. "Question 1" vs "Q1")
        def normalize(s):
            return s.lower().replace(" ", "").replace("question", "").replace("q", "")
            
        target = normalize(question)
        matched_rubric_key = next((k for k in rubric_data.keys() if normalize(k) == target), None)
        
        if not matched_rubric_key:
            results.append(QuestionResult(
                question=question, score=0.0, confidence=0.0, breakdown=[],
                matched_concepts=[], missing_concepts=[]
            ))
            continue
            
        rubrics_for_q = rubric_data[matched_rubric_key]
        rubric_points_raw = [item['point'] for item in rubrics_for_q]
        rubric_texts = [preprocess_text(p) for p in rubric_points_raw]
        rubric_weights = [item.get('weight', 1.0) for item in rubrics_for_q]
        
        if not rubric_texts:
            results.append(QuestionResult(
                question=question, score=0.0, confidence=0.0, breakdown=[],
                matched_concepts=[], missing_concepts=[]
            ))
            continue
            
        # Get embeddings concurrently
        student_emb = (await run_in_threadpool(get_embeddings, [clean_student_answer]))[0]
        rubric_embs = await run_in_threadpool(get_embeddings, rubric_texts)
        
        similarities = await run_in_threadpool(calculate_similarity, student_emb, rubric_embs)
        final_score, confidence = calculate_final_score(similarities, rubric_weights)
        
        # Determine matched and missing concepts based on a threshold (e.g. 0.6)
        matched = [rubric_points_raw[i] for i, s in enumerate(similarities) if s >= 0.6]
        missing = [rubric_points_raw[i] for i, s in enumerate(similarities) if s < 0.6]
        
        results.append(QuestionResult(
            question=question, 
            score=final_score, 
            confidence=confidence, 
            breakdown=similarities,
            matched_concepts=matched,
            missing_concepts=missing
        ))

    return GradeResponse(student_id=student_id, questions=results)
