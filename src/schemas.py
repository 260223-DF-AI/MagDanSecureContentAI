from pydantic import BaseModel

# =========================
# ERD Dimension Tables
# =========================

class DimUserSchema(BaseModel):
    """
    Represents dim_user table in ERD
    """
    user_id: str
    username: str
    num_of_posts: int
    num_of_violations: int


class DimImageSchema(BaseModel):
    """
    Represents dim_images table
    """
    image_id: str
    image_path: str
    correct_cat: str | None = None


class DimDescriptionSchema(BaseModel):
    """
    Represents dim_descriptions table
    """
    description_id: str
    text: str
    is_safe_content: bool


class DimPostSchema(BaseModel):
    """
    Represents dim_posts table
    (joins user, image, description)
    """
    post_id: str
    status: str
    user_key: str
    image_key: str
    description_key: str


# =========================
# Training / Model Output Tables
# =========================

class CNNTrainingSchema(BaseModel):
    """
    Represents CNN_Training table
    Output from image classification model
    """
    cnn_train_id: str
    confidence_score: float
    classification_cat: str
    is_correct: bool | None = None
    accuracy: float | None = None
    image_key: str


class LLMTrainingSchema(BaseModel):
    """
    Represents LLM_Training table
    Output from text reasoning model
    """
    llm_train_id: str
    output: str
    is_correct: bool | None = None
    accuracy: float | None = None
    description_key: str


class FinalModelLogsSchema(BaseModel):
    """
    Represents Final_Model_Logs table
    Final combined decision after CNN + LLM
    """
    log_id: str
    model_output: str
    is_correct_class: bool | None = None
    is_correct_prompt: bool | None = None
    is_post_allowed: bool
    policy_check_accuracy: float | None = None
    post_key: str

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