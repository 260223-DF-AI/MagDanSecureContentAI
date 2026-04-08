from fastapi import APIRouter, File, Form, UploadFile

from models.schemas import AnalyzePostResponse
from src.core.security import sanitize_user_text
from src.services.pipeline import SecureContentPipeline
from src.services.reasoning_stub import StubLLMReasoningEngine
from src.services.vision_stub import StubCNNClassifier

# Router groups all endpoints together
router = APIRouter()

# --- Service Initialization ---
# These are STUBS (temporary implementations)
# Later: replace with SageMaker / real model endpoints
cnn_service = StubCNNClassifier()
llm_service = StubLLMReasoningEngine()

# Pipeline orchestrates the full workflow (CNN -> LLM -> final output)
pipeline = SecureContentPipeline(
    cnn_classifier=cnn_service,
    llm_reasoning_engine=llm_service,
)


@router.get("/health")
async def health():
    """
    Simple health check endpoint.
    Used to verify the API is running.
    """
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalyzePostResponse)
async def analyze(
    username: str = Form(...),
    text: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Main endpoint for SecureContent AI.

    INPUT:
    - username → maps to dim_user
    - text → maps to dim_descriptions
    - file → maps to dim_images

    PROCESS:
    1. Read file
    2. Sanitize text (prevent prompt injection)
    3. Run pipeline (CNN + LLM)

    OUTPUT:
    - ERD-aligned structured response (user, post, logs, etc.)
    """
    # Read uploaded file into memory
    file_bytes = await file.read()
    # Sanitize text BEFORE sending to LLM (security requirement)
    sanitized_text = sanitize_user_text(text)

    # Run full moderation pipeline
    result = await pipeline.run(
        username=username,
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        mime_type=file.content_type or "application/octet-stream",
        original_text=text,
        sanitized_text=sanitized_text,
    )

    return result