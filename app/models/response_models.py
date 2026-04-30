from pydantic import BaseModel
from typing import List, Optional

class QuestionResult(BaseModel):
    question: str
    score: float
    confidence: float
    breakdown: List[float]

class GradeResponse(BaseModel):
    student_id: Optional[str] = "UNKNOWN"
    questions: List[QuestionResult]

class BatchJobResponse(BaseModel):
    job_id: str
    message: str

class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    results: List[GradeResponse] = []
