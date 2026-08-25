# src/ai-service/app/models/request_models.py
from pydantic import BaseModel
from typing import List, Optional

class SimilarityRequest(BaseModel):
    student_answer: str
    rubric: List[str]

class SegmentRequest(BaseModel):
    raw_text: str
    # Optional canonical rubric labels (e.g. ["1", "2a", "3"]). When supplied,
    # segmentation is rubric-aware: only real questions are detected and answers
    # can no longer merge or fragment on unrecognised labelling styles.
    expected_labels: Optional[List[str]] = None
