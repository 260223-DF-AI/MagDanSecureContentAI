# creates the package model group, then registers the trained model
# as a version in sage model registry

import os

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.pytorch import PyTorchModel


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "REPLACE_ME")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "REPLACE_ME")
MODEL_PACKAGE_GROUP_NAME = os.getenv(
    "MODEL_PACKAGE_GROUP_NAME",
    "securecontent-cnn-model-group",
)

# Paste the actual artifact path from launch_training.py output if needed
MODEL_ARTIFACT_S3_URI = os.getenv(
    "MODEL_ARTIFACT_S3_URI",
    "REPLACE_ME",
)

FRAMEWORK_VERSION = "2.4.0"
PY_VERSION = "py311"


def ensure_model_package_group(sm_client, group_name: str) -> None:
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"Model package group already exists: {group_name}")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "ValidationException":
            sm_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription="Versioned CNN moderation models for SecureContent AI",
            )
            print(f"Created model package group: {group_name}")
        else:
            raise


def main() -> None:
    boto_session = boto3.Session(region_name=AWS_REGION)
    sm_client = boto_session.client("sagemaker")
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=S3_BUCKET,
    )

    ensure_model_package_group(sm_client, MODEL_PACKAGE_GROUP_NAME)

    # No custom inference script needed yet for registry.
    # This uses the SageMaker PyTorch inference container metadata.
    pytorch_model = PyTorchModel(
        model_data=MODEL_ARTIFACT_S3_URI,
        role=ROLE_ARN,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        sagemaker_session=sagemaker_session,
    )

    model_package_arn = pytorch_model.register(
        content_types=["application/x-image"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.c5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        description="SecureContent AI CNN moderation model",
    )

    print("Registered model package.")
    print(f"Model package ARN: {model_package_arn}")


if __name__ == "__main__":
    main()