from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SecureContent AI"
    app_version: str = "0.1.0"
    debug: bool = True

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/securecontent_ai"
    )

    # Placeholder for later SageMaker integration
    sagemaker_endpoint_name: str | None = None
    sagemaker_model_package_group: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
