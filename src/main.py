from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.models.schemas import AnalyzeResponse
from src.services.vision_service import VisionService
from src.services.llm_service import LLMService

app = FastAPI(
    title="SecureContent AI",
    description="FastAPI service connecting a vision classifier and LLM moderation reasoning.",
    version="1.0.0",
)

vision_service = VisionService()
llm_service = LLMService()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "SecureContent AI API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_content(
    file: Annotated[UploadFile, File(...)],
    text: Annotated[str, Form(...)],
) -> AnalyzeResponse:
    """
    Accepts:
    - one uploaded image file
    - one text field

    Returns:
    - vision result
    - llm moderation result
    - combined final decision
    """
    supported_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported types: {sorted(supported_types)}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        vision_result = vision_service.analyze_image(
            image_bytes=image_bytes,
            filename=file.filename or "uploaded_file",
            content_type=file.content_type,
        )

        llm_result = llm_service.analyze_text(
            text=text,
            vision_result=vision_result,
        )

        final_allowed = vision_result["is_post_allowed"] and llm_result["is_comment_allowed"]

        combined_reason = {
            "vision_summary": vision_result["reason"],
            "llm_summary": llm_result["reason"],
            "final_reason": (
                "Approved" if final_allowed else "Blocked due to image classification and/or text policy review."
            ),
        }

        return AnalyzeResponse(
            filename=file.filename or "uploaded_file",
            content_type=file.content_type,
            input_text=text,
            vision=vision_result,
            llm=llm_result,
            final_decision="approved" if final_allowed else "blocked",
            is_allowed=final_allowed,
            reasoning=combined_reason,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc