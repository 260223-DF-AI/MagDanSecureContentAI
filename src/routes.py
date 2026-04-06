from fastapi import APIRouter, File, Form, UploadFile

from src.schemas import AnalyzeResponse
from src.services.pipeline import ModerationPipeline
from src.services.vision_stub import StubVisionClassifier
from src.services.reasoning_stub import StubReasoningEngine
from src.security import sanitize_user_text

router = APIRouter()

# Plug in stub services (replace later with SageMaker)
vision_service = StubVisionClassifier()
reasoning_service = StubReasoningEngine()

pipeline = ModerationPipeline(
    vision_classifier=vision_service,
    reasoning_engine=reasoning_service,
)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    text: str = Form(...),
    file: UploadFile = File(...),
):
    file_bytes = await file.read()

    sanitized_text = sanitize_user_text(text)

    result = await pipeline.run(
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "application/octet-stream",
        user_text=sanitized_text,
    )

    return result