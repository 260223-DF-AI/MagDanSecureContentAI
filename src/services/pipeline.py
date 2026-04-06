from uuid import uuid4

from src.schemas import (
    AnalyzePostResponse,
    CNNTrainingSchema,
    DimDescriptionSchema,
    DimImageSchema,
    DimPostSchema,
    DimUserSchema,
    FinalModelLogsSchema,
    LLMTrainingSchema,
)
from src.services.interfaces import CNNClassifier, LLMReasoningEngine


class SecureContentPipeline:
    def __init__(
        self,
        cnn_classifier: CNNClassifier,
        llm_reasoning_engine: LLMReasoningEngine,
    ):
        self.cnn_classifier = cnn_classifier
        self.llm_reasoning_engine = llm_reasoning_engine

    async def run(
        self,
        username: str,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        original_text: str,
        sanitized_text: str,
    ) -> AnalyzePostResponse:
        user_id = f"user-{uuid4().hex[:8]}"
        image_id = f"img-{uuid4().hex[:8]}"
        description_id = f"desc-{uuid4().hex[:8]}"
        post_id = f"post-{uuid4().hex[:8]}"
        log_id = f"log-{uuid4().hex[:8]}"

        user = DimUserSchema(
            user_id=user_id,
            username=username,
            num_of_posts=0,
            num_of_violations=0,
        )

        image = DimImageSchema(
            image_id=image_id,
            image_path=filename,
            correct_cat=None,
        )

        cnn_training: CNNTrainingSchema = await self.cnn_classifier.classify_image(
            file_bytes=file_bytes,
            filename=filename,
            image_id=image_id,
        )

        llm_training, trace = await self.llm_reasoning_engine.analyze_description(
            text=sanitized_text,
            description_id=description_id,
            classification_cat=cnn_training.classification_cat,
        )

        is_safe_content = llm_training.output == "Approved"
        is_post_allowed = is_safe_content
        post_status = "approved" if is_post_allowed else "policy_violation"

        description = DimDescriptionSchema(
            description_id=description_id,
            text=original_text,
            is_safe_content=is_safe_content,
        )

        post = DimPostSchema(
            post_id=post_id,
            status=post_status,
            user_key=user.user_id,
            image_key=image.image_id,
            description_key=description.description_id,
        )

        final_model_log = FinalModelLogsSchema(
            log_id=log_id,
            model_output=llm_training.output,
            is_correct_class=None,
            is_correct_prompt=None,
            is_post_allowed=is_post_allowed,
            policy_check_accuracy=None,
            post_key=post.post_id,
        )

        return AnalyzePostResponse(
            user=user,
            image=image,
            description=description,
            post=post,
            cnn_training=cnn_training,
            llm_training=llm_training,
            final_model_log=final_model_log,
            trace=trace,
        )