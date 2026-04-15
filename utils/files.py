from fastapi import HTTPException, UploadFile

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def validate_and_read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: JPEG, PNG, WEBP.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return content
