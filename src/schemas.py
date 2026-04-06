from pydantic import BaseModel


class VisionResult(BaseModel):
    label: str
    confidence: float


class ReasoningResult(BaseModel):
    decision: str
    generated_response: str | None = None
    trace: dict


class AnalyzeResponse(BaseModel):
    image_label: str
    image_confidence: float
    final_decision: str
    generated_response: str | None = None
    trace: dict