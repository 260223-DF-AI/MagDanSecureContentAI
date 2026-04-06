from fastapi import APIRouter, File, Form, UploadFile

from src.schemas import AnalyzePostResponse
from src.core.security import sanitize_user_text
from src.services.pipeline import SecureContentPipeline
from src.services.reasoning_stub import StubLLMReasoningEngine
from src.services.vision_stub import StubCNNClassifier

router = APIRouter()

cnn_service = StubCNNClassifier()
llm_service = StubLLMReasoningEngine()

pipeline = SecureContentPipeline(
    cnn_classifier=cnn_service,
    llm_reasoning_engine=llm_service,
)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzePostResponse)
async def analyze(
    username: str = Form(...),
    text: str = Form(...),
    file: UploadFile = File(...),
):
    file_bytes = await file.read()
    sanitized_text = sanitize_user_text(text)

    result = await pipeline.run(
        username=username,
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "application/octet-stream",
        original_text=text,
        sanitized_text=sanitized_text,
    )

    return result