# script that starts the remote training job

import os

import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch


# -------------------------
# Fill these in
# -------------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "REPLACE_ME")
S3_PREFIX = os.getenv("S3_PREFIX", "securecontent-ai")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "REPLACE_ME")

TRAIN_S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/train"
TEST_S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX}/data/test"
OUTPUT_S3_URI = f"s3://{S3_BUCKET}/{S3_PREFIX}/output"

FRAMEWORK_VERSION = "2.4.0"
PY_VERSION = "py311"
INSTANCE_TYPE = "ml.g4dn.xlarge"  # use a GPU if available; change if needed
INSTANCE_COUNT = 1


def main() -> None:
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(
        boto_session=boto_session,
        default_bucket=S3_BUCKET,
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=".",  # assumes train.py is at repo root
        role=ROLE_ARN,
        framework_version=FRAMEWORK_VERSION,
        py_version=PY_VERSION,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        output_path=OUTPUT_S3_URI,
        hyperparameters={
            "epochs": 10,
            "batch_size": 16,
            "lr": 0.001,
            "patience": 2,
            "num_workers": 2,
            "use_amp": True,
        },
        base_job_name="securecontent-cnn-train",
    )

    estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=TRAIN_S3_URI,
                content_type="application/x-image",
            ),
            "test": TrainingInput(
                s3_data=TEST_S3_URI,
                content_type="application/x-image",
            ),
        },
        wait=True,
        logs=True,
    )

    print("Training job complete.")
    print(f"Latest training job name: {estimator.latest_training_job.name}")
    print(f"Model artifact S3 URI: {estimator.model_data}")


if __name__ == "__main__":
    main()