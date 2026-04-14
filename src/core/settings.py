import os


class Settings:
    app_name: str = "SecureContent AI"
    env: str = os.getenv("ENV", "dev")
    vision_endpoint_name: str = os.getenv("VISION_ENDPOINT_NAME", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")


settings = Settings()