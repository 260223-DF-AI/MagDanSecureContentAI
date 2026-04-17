from pydantic import BaseModel
from typing import Any

# validates data types using pydantic

# =========================
# ERD Dimension Tables
# =========================

class DimUserSchema(BaseModel):
    """
    Represents dim_user table in ERD
    """
    user_id: int | None = None
    username: str
    num_of_posts: int
    num_of_violations: int


class DimImageSchema(BaseModel):
    """
    Represents dim_images table
    """
    image_id: int | None = None
    image_path: str
    label: str | None = None


class DimDescriptionSchema(BaseModel):
    """
    Represents dim_descriptions table
    """
    description_id: int | None = None
    text: str
    is_safe_content: bool | None = None


class DimPostSchema(BaseModel):
    """
    Represents dim_posts table
    (joins user, image, description)
    """
    post_id: int | None = None
    status: str | None = None
    user_key: int
    image_key: int
    description_key: int


# =========================
# Training / Model Output Tables
# =========================

class CNNTrainingSchema(BaseModel):
    """
    Represents CNN_Training table
    Output from image classification model
    """
    cnn_train_id: int | None = None
    confidence_score: float
    predicted_class: str
    is_correct: bool | None = None
    image_key: int
    run_key: int


class LLMTrainingSchema(BaseModel):
    """
    Represents LLM_Training table
    Output from text reasoning model
    """
    llm_train_id: int | None = None
    reasoning: str
    moderation_decision: str
    is_correct: bool | None = None
    description_key: int


class FinalModelLogsSchema(BaseModel):
    """
    Represents Final_Model_Logs table
    Final combined decision after CNN + LLM
    """
    log_id: int | None = None
    model_output: str
    is_correct_class: bool | None = None
    is_correct_prompt: bool | None = None
    is_post_allowed: bool
    policy_check_accuracy: float | None = None
    post_key: int

class VisionResult(BaseModel):
    predicted_class: str
    confidence_score: float
    class_probabilities: dict[str, float]
    is_post_allowed: bool
    reason: str

class LLMResult(BaseModel):
    moderation_label: str
    is_comment_allowed: bool
    reason: str
    suggested_response: str

# =========================
# Final API Response
# =========================

class AnalyzeResponse(BaseModel):
    """
    Full API response — mirrors ERD structure.

    This is what /analyze returns.
    """
    filename: str
    content_type: str
    input_text: str
    vision: VisionResult
    llm: LLMResult
    final_decision: str
    is_allowed: bool
    reasoning: dict[str, Any]