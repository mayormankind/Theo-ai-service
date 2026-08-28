# src/ai-service/app/routes/grading.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
import json
import os
import re
from fastapi.concurrency import run_in_threadpool

from app.models.response_models import GradeResponse, QuestionResult
from app.services.ocr_service import extract_text_hybrid
from app.services.segmentation_service import segment_answers
from app.services.embedding_service import get_embeddings
from app.services.embeddings import EmbeddingError
from app.services.scoring_service import calculate_similarity, calculate_final_score
from app.utils.text_preprocessing import preprocess_text, extract_student_id, clean_ocr_output

router = APIRouter()

_QUESTION_PREFIX_RE = re.compile(r'^(?:question\s*|q)', re.IGNORECASE)

def normalize_question_label(s: str) -> str:
    """
    Normalise a question label to a bare digit string for matching.
    e.g. "Question 1" → "1", "Q2b" → "2b", "2(a)" → "2a", "2." → "2"
    Only the leading 'question' or 'q' prefix is stripped, not every 'q'
    in the string (which would corrupt words like 'equal').
    """
    s = s.lower().strip()
    s = _QUESTION_PREFIX_RE.sub('', s)
    s = re.sub(r'[\s\.\(\)]', '', s)
    return s


@router.post("/grade", response_model=GradeResponse)
async def grade_endpoint(
    rubric_str: str = Form(...),
    file: Optional[UploadFile] = File(None),
    extracted_text: Optional[str] = Form(None),
):
    """
    Single script grading pipeline.
    """
    try:
        rubric_data = json.loads(rubric_str)
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail="Invalid rubric JSON format"
        )

    # Use pre-extracted text if provided, 
    # otherwise run OCR on the file
    if extracted_text and extracted_text.strip():
        raw_text = clean_ocr_output(extracted_text.strip())
        student_id = extract_student_id(raw_text)
    elif file:
        image_bytes = await file.read()
        filename = file.filename or "image.jpg"
        ocr_result = await run_in_threadpool(
            extract_text_hybrid, image_bytes, filename
        )
        raw_text = clean_ocr_output(ocr_result.get("extracted_text", ""))
        student_id = extract_student_id(raw_text)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either file or extracted_text required"
        )

    if not raw_text:
        return GradeResponse(
            student_id="UNKNOWN", questions=[]
        )

    from app.config.constants import (
        SIMILARITY_FULL, SIMILARITY_PARTIAL
    )

    # Step 2: Rubric-aware segmentation.
    # Passing the rubric's canonical labels makes segmentation deterministic and
    # stops answers from merging/fragmenting on unusual labelling styles.
    expected_labels = [normalize_question_label(k) for k in rubric_data.keys()]
    segmented_answers = segment_answers(raw_text, expected_labels=expected_labels)

    # Map canonical label -> (document_order, answer_text). Dict insertion order
    # from the segmenter is document order (top-to-bottom scan).
    seg_lookup = {}
    for i, (k, v) in enumerate(segmented_answers.items()):
        seg_lookup[normalize_question_label(k)] = (i, v)

    # Emit answered questions first (in document order) then unanswered ones, so
    # the array index still reflects document order for FIRST_N selection.
    rubric_keys = list(rubric_data.keys())

    def _order_key(rk: str):
        t = normalize_question_label(rk)
        if t in seg_lookup:
            return (0, seg_lookup[t][0])
        return (1, rubric_keys.index(rk))

    ordered_keys = sorted(rubric_keys, key=_order_key)

    results = []

    # Iterate over the RUBRIC, not the segments, so EVERY question always gets a
    # result with its correct max marks. A question the student didn't answer
    # (or that couldn't be located) scores 0 instead of silently vanishing —
    # which previously produced misleading "0/60, only 1 of 3 graded" reports.
    for question in ordered_keys:
        rubrics_for_q = rubric_data[question] or []
        target = normalize_question_label(question)
        answer_text = (seg_lookup.get(target, (0, ""))[1] or "").strip()

        rubric_points_raw = [item['point'] for item in rubrics_for_q]
        rubric_texts = [p.strip() for p in rubric_points_raw]
        rubric_weights = [item.get('weight', 1.0) for item in rubrics_for_q]

        # Get max marks for this question. Prefer explicit questionMaxScore,
        # fallback to the sum of point maxScores. Guard against a question that
        # was defined with no rubric points.
        question_max = (
            rubrics_for_q[0].get(
                'questionMaxScore',
                sum(item.get('maxScore', 0) for item in rubrics_for_q)
            )
            if rubrics_for_q else 0
        )

        # No answer located, or no rubric points: emit an explicit zero so the
        # question is still accounted for in the total.
        if not answer_text or not rubric_texts:
            results.append(QuestionResult(
                question=question,
                answer=answer_text,
                score=0.0,
                confidence=0.0,
                breakdown=[],
                matched_concepts=[],
                partial_concepts=[],
                missing_concepts=rubric_points_raw,
            ))
            continue

        clean_student_answer = answer_text

        # Get embeddings concurrently.
        # If the embedding provider fails, abort the whole request with 503
        # rather than letting a zero-vector silently score the student 0.
        try:
            student_emb = (await run_in_threadpool(get_embeddings, [clean_student_answer]))[0]
            rubric_embs = await run_in_threadpool(get_embeddings, rubric_texts)
        except EmbeddingError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Embedding provider unavailable — script not graded: {e}"
            )

        similarities = await run_in_threadpool(calculate_similarity, student_emb, rubric_embs)
        final_score, confidence = calculate_final_score(
            similarities,
            rubric_weights,
            question_max_score=question_max
        )

        print(f"[GRADE] Q:{question} | "
              f"max:{question_max} | "
              f"sim:{[round(s,2) for s in similarities]} | "
              f"score:{final_score}")

        # Determine matched, partial and missing concepts based on aligned thresholds
        matched = [rubric_points_raw[i] for i, s in
                   enumerate(similarities)
                   if s >= SIMILARITY_FULL]
        partial = [rubric_points_raw[i] for i, s in
                   enumerate(similarities)
                   if SIMILARITY_PARTIAL <= s < SIMILARITY_FULL]
        missing = [rubric_points_raw[i] for i, s in
                   enumerate(similarities)
                   if s < SIMILARITY_PARTIAL]

        results.append(QuestionResult(
            question=question,
            answer=answer_text,
            score=final_score,
            confidence=confidence,
            breakdown=similarities,
            matched_concepts=matched,
            partial_concepts=partial,
            missing_concepts=missing
        ))

    return GradeResponse(student_id=student_id, questions=results)
