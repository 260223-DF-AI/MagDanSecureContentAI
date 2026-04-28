from typing import Annotated
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import mimetypes
from pathlib import Path
from html import escape
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from src.models.schemas import AnalyzeResponse
from src.services.vision_service import VisionService
from src.services.llm_service import LLMService
from dotenv import load_dotenv
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        cursor_factory=RealDictCursor,
    )

app = FastAPI(
    title="SecureContent AI",
    description="FastAPI service connecting a vision classifier and LLM moderation reasoning.",
    version="1.0.0",
)

vision_service = VisionService()
llm_service = LLMService()

# helper methods to look up image paths in the database
def get_post_by_id(post_id: int):
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    p.post_id,
                    p.status,
                    p.user_key,
                    p.image_key,
                    p.description_key,
                    d.text AS description_text,
                    d.is_safe_content,
                    u.username,
                    u.num_of_posts,
                    u.num_of_violations,
                    i.image_path,
                    i.label
                FROM dim_post AS p
                INNER JOIN dim_user AS u
                    ON p.user_key = u.user_id
                INNER JOIN dim_description AS d
                    ON p.description_key = d.description_id
                INNER JOIN dim_image AS i
                    ON p.image_key = i.image_id
                WHERE p.post_id = %s
                """,
                (post_id,),
            )
            return cursor.fetchone()
    finally:
        conn.close()


def resolve_image_file(raw_path: str) -> Path:
    """
    Converts a DB image path into a real local file path.
    Works for values like:
    - /imagesDemo/file.jpg
    - imagesDemo/file.jpg
    - /absolute/path/file.jpg
    """
    if not raw_path:
        raise FileNotFoundError("No image path found for post.")

    raw = Path(raw_path)

    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    cleaned = raw_path.lstrip("/")

    # current working directory
    candidates.append(Path.cwd() / cleaned)

    # same folder as main.py
    base_dir = Path(__file__).resolve().parent
    candidates.append(base_dir / cleaned)

    # repo/website/imagesDemo style
    candidates.append(base_dir / "website" / cleaned)
    candidates.append(base_dir.parent / "website" / cleaned)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find image file for path: {raw_path}")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "SecureContent AI API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

# verify button in the feed 
@app.get("/verify/{post_id}", response_class=HTMLResponse)
async def verify_post(post_id: int):
    try:
        post = get_post_by_id(post_id)
        if not post:
          raise HTTPException(status_code=404, detail="Post not found.")
        
        image_path = post["image_path"]
        if not image_path:
            raise HTTPException(status_code=400, detail="Post has no image path")

        image_file = resolve_image_file(post["image_path"])
        content_type, _ = mimetypes.guess_type(str(image_file))
        content_type = content_type or "image/jpeg"

        with open(image_file, "rb") as f:
            image_bytes = f.read()

        vision_result = vision_service.analyze_image(
            image_bytes=image_bytes,
            filename=image_file.name,
            content_type=content_type,
        )

        llm_result = llm_service.analyze_text(
            text=post["description_text"],
            vision_result=vision_result,
        )

        final_allowed = (
            vision_result["is_post_allowed"]
            and llm_result["is_comment_allowed"]
        )

        status_text = "APPROVED" if final_allowed else "BLOCKED"
        status_class = "approved" if final_allowed else "blocked"

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Verify Post #{post_id}</title>
          <style>
            body {{
              font-family: Arial, Helvetica, sans-serif;
              background: linear-gradient(180deg, #fff8fc 0%, #ffeef7 100%);
              color: #5b2d49;
              margin: 0;
              padding: 32px;
            }}
            .page {{
              max-width: 900px;
              margin: 0 auto;
            }}
            .back {{
              display: inline-block;
              margin-bottom: 18px;
              padding: 10px 18px;
              border-radius: 999px;
              text-decoration: none;
              font-weight: 700;
              background: white;
              color: #de3b92;
              box-shadow: 0 8px 18px rgba(242, 93, 168, 0.12);
            }}
            .card {{
              background: rgba(255,255,255,0.82);
              border-radius: 28px;
              padding: 28px;
              box-shadow: 0 18px 40px rgba(222, 59, 146, 0.14);
              border: 1px solid rgba(242, 93, 168, 0.18);
            }}
            .status {{
              display: inline-block;
              padding: 10px 16px;
              border-radius: 999px;
              font-weight: 800;
              margin-bottom: 18px;
            }}
            .approved {{
              background: linear-gradient(135deg, #9bf3c0, #39c97a);
              color: white;
            }}
            .blocked {{
              background: linear-gradient(135deg, #ff9fb6, #ff5d88);
              color: white;
            }}
            h1 {{
              margin-top: 0;
            }}
            .section {{
              margin-top: 22px;
              padding: 18px;
              border-radius: 20px;
              background: rgba(255,255,255,0.7);
            }}
            .label {{
              font-weight: 800;
              color: #de3b92;
              margin-bottom: 8px;
            }}
            .meta {{
              color: #7f4d68;
              line-height: 1.7;
            }}
            .reason {{
              line-height: 1.7;
            }}
          </style>
        </head>
        <body>
          <div class="page">
            <a class="back" href="http://localhost:3000/feed.html">← Back to Feed</a>

            <div class="card">
              <div class="status {status_class}">{status_text}</div>
              <h1>Verification Result for Post #{post_id}</h1>

              <div class="section">
                <div class="label">Post Info</div>
                <div class="meta">
                  <div><strong>User:</strong> {escape(str(post["username"]))}</div>
                  <div><strong>Status in DB:</strong> {escape(str(post["status"]))}</div>
                  <div><strong>Image Label:</strong> {escape(str(post["label"]))}</div>
                  <div><strong>Description:</strong> {escape(str(post["description_text"]))}</div>
                </div>
              </div>

              <div class="section">
                <div class="label">Vision Result</div>
                <div class="reason">
                  <div><strong>Allowed:</strong> {vision_result["is_post_allowed"]}</div>
                  <div><strong>Reason:</strong> {escape(str(vision_result["reason"]))}</div>
                </div>
              </div>

              <div class="section">
                <div class="label">LLM Result</div>
                <div class="reason">
                  <div><strong>Allowed:</strong> {llm_result["is_comment_allowed"]}</div>
                  <div><strong>Reason:</strong> {escape(str(llm_result["reason"]))}</div>
                </div>
              </div>

              <div class="section">
                <div class="label">Final Decision</div>
                <div class="reason">
                  {"Approved" if final_allowed else "Blocked due to image classification and/or text policy review."}
                </div>
              </div>
            </div>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verify failed: {exc}") from exc
    
# to "create a post" - to import and image and type in a description string for it to analyze
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
