from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import json
import uuid
import asyncio
from typing import List
from fastapi.concurrency import run_in_threadpool

from app.models.response_models import GradeResponse, QuestionResult, BatchJobResponse, BatchJobStatus
from app.services.ocr_service import extract_text_hybrid
from app.services.segmentation_service import segment_answers
from app.services.embedding_service import get_embeddings
from app.services.scoring_service import calculate_similarity, calculate_final_score
from app.utils.text_preprocessing import preprocess_text, extract_student_id

router = APIRouter()

# Global mock DB for jobs (would be Redis/DB in production)
JOB_STORE = {}

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
        
        matched_rubric_key = next((k for k in rubric_data.keys() if k.lower().strip() == question.lower().strip()), None)
        if not matched_rubric_key:
            results.append(QuestionResult(question=question, score=0.0, confidence=0.0, breakdown=[]))
            continue
            
        rubrics_for_q = rubric_data[matched_rubric_key]
        rubric_texts = [preprocess_text(item['point']) for item in rubrics_for_q]
        rubric_weights = [item.get('weight', 1.0) for item in rubrics_for_q]
        
        if not rubric_texts:
            results.append(QuestionResult(question=question, score=0.0, confidence=0.0, breakdown=[]))
            continue
            
        # Get embeddings concurrently
        student_emb = (await run_in_threadpool(get_embeddings, [clean_student_answer]))[0]
        rubric_embs = await run_in_threadpool(get_embeddings, rubric_texts)
        
        similarities = await run_in_threadpool(calculate_similarity, student_emb, rubric_embs)
        final_score, confidence = calculate_final_score(similarities, rubric_weights)
        
        results.append(QuestionResult(question=question, score=final_score, confidence=confidence, breakdown=similarities))

    return GradeResponse(student_id=student_id, questions=results)

@router.post("/batch-grade", response_model=BatchJobResponse)
async def batch_grade_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    rubric_str: str = Form(...)
):
    try:
        rubric_data = json.loads(rubric_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid rubric JSON format")
        
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "processing", "results": []}
    
    # Pre-compute rubric embeddings ONCE for the whole batch
    precomputed_rubrics = {}
    for question_key, rubrics_for_q in rubric_data.items():
        rubric_texts = [preprocess_text(item['point']) for item in rubrics_for_q]
        rubric_weights = [item.get('weight', 1.0) for item in rubrics_for_q]
        if rubric_texts:
            embs = await run_in_threadpool(get_embeddings, rubric_texts)
            precomputed_rubrics[question_key.lower().strip()] = {
                "texts": rubric_texts,
                "weights": rubric_weights,
                "embeddings": embs
            }
            
    file_bytes_list = []
    for f in files:
        file_bytes_list.append((await f.read(), f.filename or "image.jpg"))
        
    background_tasks.add_task(process_batch_grading, job_id, file_bytes_list, precomputed_rubrics)
    
    return BatchJobResponse(job_id=job_id, message="Batch grading started.")

@router.get("/batch-status/{job_id}", response_model=BatchJobStatus)
def get_batch_status(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return BatchJobStatus(job_id=job_id, status=job["status"], results=job["results"])

async def process_batch_grading(job_id: str, file_list: list, precomputed_rubrics: dict):
    results = []
    for file_bytes, filename in file_list:
        try:
            ocr_result = await run_in_threadpool(extract_text_hybrid, file_bytes, filename)
            raw_text = ocr_result.get("extracted_text", "")
            if not raw_text:
                continue
            student_id = extract_student_id(raw_text)
            segmented_answers = segment_answers(raw_text)
            question_results = []
            
            for question, answer_text in segmented_answers.items():
                if not answer_text.strip():
                    question_results.append(QuestionResult(question=question, score=0.0, confidence=0.0, breakdown=[]))
                    continue
                clean_student_answer = preprocess_text(answer_text)
                matched_key = question.lower().strip()
                rubric_info = precomputed_rubrics.get(matched_key)
                
                if not rubric_info:
                    question_results.append(QuestionResult(question=question, score=0.0, confidence=0.0, breakdown=[]))
                    continue
                    
                student_emb = (await run_in_threadpool(get_embeddings, [clean_student_answer]))[0]
                rubric_embs = rubric_info["embeddings"]
                rubric_weights = rubric_info["weights"]
                
                similarities = await run_in_threadpool(calculate_similarity, student_emb, rubric_embs)
                final_score, confidence = calculate_final_score(similarities, rubric_weights)
                
                question_results.append(QuestionResult(question=question, score=final_score, confidence=confidence, breakdown=similarities))
            results.append(GradeResponse(student_id=student_id, questions=question_results))
        except Exception as e:
             print(f"Failed to process file in batch: {e}")
             
    JOB_STORE[job_id]["results"] = results
    JOB_STORE[job_id]["status"] = "completed"
