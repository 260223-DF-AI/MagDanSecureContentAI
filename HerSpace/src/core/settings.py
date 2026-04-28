import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    app_name: str = "SecureContent AI"
    env: str = os.getenv("ENV", "dev")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    # fastapi will call this SageMaker endpoint for vision inference ---- add in after deployment (this gives you the value)
    VISION_ENDPOINT_NAME = os.getenv("VISION_ENDPOINT_NAME")
    vision_endpoint_name: str = VISION_ENDPOINT_NAME
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
settings = Settings()