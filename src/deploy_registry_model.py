# this is the model that is like "classifier.py"
# or more specifically, the image classifier model.
# it is stored in the SageMaker Registry, this program is to deploy it
# for use in the SageMaker endpoint, and FastAPI

import os
import time
import boto3
import sagemaker
from sagemaker.model import ModelPackage

AWS_REGION = os.getenv("AWS_REGION")
MODEL_PACKAGE_ARN = os.getenv("MODEL_PACKAGE_ARN")
ROLE_ARN = os.getenv("ROLE_ARN")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.m5.large")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
if not ENDPOINT_NAME:
    ENDPOINT_NAME = f"securecontent-vision-endpoint-{int(time.time())}"
def validate_env():
    missing = []

    if not AWS_REGION:
        missing.append("AWS_REGION")
    if not MODEL_PACKAGE_ARN:
        missing.append("MODEL_PACKAGE_ARN")
    if not ROLE_ARN:
        missing.append("ROLE_ARN")

    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")


def main() -> None:
    validate_env()

    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    print("Region:", AWS_REGION)
    print("Endpoint name:", ENDPOINT_NAME)
    print("Model package ARN:", MODEL_PACKAGE_ARN)

    # Deploy approved model package from Model Registry
    model = ModelPackage(
        role=ROLE_ARN,
        model_package_arn=MODEL_PACKAGE_ARN,
        sagemaker_session=sagemaker_session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print("Deployment started / completed.")
    print("Endpoint name:", predictor.endpoint_name)

    # helpful for FastAPI integration
    print("\n👉 Add this to your .env for FastAPI:")
    print(f"VISION_ENDPOINT_NAME={predictor.endpoint_name}")


if __name__ == "__main__":
    main()