from pydantic import BaseModel
from typing import List

class SimilarityRequest(BaseModel):
    student_answer: str
    rubric: List[str]

class SegmentRequest(BaseModel):
    raw_text: str
