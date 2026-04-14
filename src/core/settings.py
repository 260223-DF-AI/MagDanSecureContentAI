import os


class Settings:
    app_name: str = "SecureContent AI"
    env: str = os.getenv("ENV", "dev")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    # fastapi will call this SageMaker endpoint for vision inference
    vision_endpoint_name: str = os.getenv("VISION_ENDPOINT_NAME", "")
    # optional: keep llm provider configurable
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")


settings = Settings()