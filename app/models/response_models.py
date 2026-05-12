from pydantic import BaseModel
from typing import List, Optional

class QuestionResult(BaseModel):
    question: str
    score: float
    confidence: float
    breakdown: List[float]
    matched_concepts: List[str] = []
    missing_concepts: List[str] = []

class GradeResponse(BaseModel):
    student_id: Optional[str] = "UNKNOWN"
    questions: List[QuestionResult]
