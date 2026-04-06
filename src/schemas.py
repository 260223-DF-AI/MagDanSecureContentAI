from pydantic import BaseModel


class DimUserSchema(BaseModel):
    user_id: str
    username: str
    num_of_posts: int
    num_of_violations: int


class DimImageSchema(BaseModel):
    image_id: str
    image_path: str
    correct_cat: str | None = None


class DimDescriptionSchema(BaseModel):
    description_id: str
    text: str
    is_safe_content: bool


class DimPostSchema(BaseModel):
    post_id: str
    status: str
    user_key: str
    image_key: str
    description_key: str


class CNNTrainingSchema(BaseModel):
    cnn_train_id: str
    confidence_score: float
    classification_cat: str
    is_correct: bool | None = None
    accuracy: float | None = None
    image_key: str


class LLMTrainingSchema(BaseModel):
    llm_train_id: str
    output: str
    is_correct: bool | None = None
    accuracy: float | None = None
    description_key: str


class FinalModelLogsSchema(BaseModel):
    log_id: str
    model_output: str
    is_correct_class: bool | None = None
    is_correct_prompt: bool | None = None
    is_post_allowed: bool
    policy_check_accuracy: float | None = None
    post_key: str


class AnalyzePostResponse(BaseModel):
    user: DimUserSchema
    image: DimImageSchema
    description: DimDescriptionSchema
    post: DimPostSchema
    cnn_training: CNNTrainingSchema
    llm_training: LLMTrainingSchema
    final_model_log: FinalModelLogsSchema
    trace: dict