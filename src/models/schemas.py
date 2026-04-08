from pydantic import BaseModel

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
    is_safe_content: bool


class DimPostSchema(BaseModel):
    """
    Represents dim_posts table
    (joins user, image, description)
    """
    post_id: int | None = None
    status: str
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
    output: str
    is_correct: bool | None = None
    accuracy: float | None = None
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

# =========================
# Final API Response
# =========================

class AnalyzePostResponse(BaseModel):
    """
    Full API response — mirrors ERD structure.

    This is what /analyze returns.
    """
    user: DimUserSchema
    image: DimImageSchema
    description: DimDescriptionSchema
    post: DimPostSchema
    cnn_training: CNNTrainingSchema
    llm_training: LLMTrainingSchema
    final_model_log: FinalModelLogsSchema
    # Debug / explainability trace (ReAct reasoning)
    trace: dict