import os
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch
from sagemaker.serializers import IdentitySerializer
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
FRAMEWORK_VERSION = "2.4.0"
PY_VERSION = "py311"

USE_GPU = False
INSTANCE_TYPE = "ml.g4dn.xlarge" if USE_GPU else 'ml.m5.large'  # use a GPU if available; change if needed
INSTANCE_COUNT = 1

ROLE_ARN = sagemaker.get_execution_role()  # This will only work if running in SageMaker environment with proper permissions
print(ROLE_ARN)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
session = sagemaker.Session()

def get_content_type_from_key(key: str) -> str:
    key = key.lower()
    if key.endswith(".jpg") or key.endswith(".jpeg"):
        return "image/jpeg"
    if key.endswith(".png"):
        return "image/png"
    if key.endswith(".webp"):
        return "image/webp"
    raise ValueError(f"Unsupported image type for key: {key}")

def predict_s3_folder(predictor, bucket: str, prefix: str) -> list[dict]:
    """
    Walk through all images in an S3 folder, send each image to the endpoint,
    and collect the prediction results.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    results = []

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if not key.lower().endswith(IMAGE_EXTENSIONS):
                continue

            content_type = get_content_type_from_key(key)

            # Update serializer to match the image type
            predictor.serializer = IdentitySerializer(content_type=content_type)

            image_obj = s3.get_object(Bucket=bucket, Key=key)
            image_bytes = image_obj["Body"].read()

            response = predictor.predict(image_bytes)

            result = {
                "s3_key": key,
                "predicted_class": response.get("predicted_class"),
                "confidence_score": response.get("confidence_score"),
                "class_probabilities": response.get("class_probabilities"),
                "is_post_allowed": response.get("is_post_allowed"),
            }

            results.append(result)
            print(result)

    return results

def main() -> None:
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # Let SageMaker create/use its default bucket
    bucket = sagemaker_session.default_bucket()
    print("Using bucket:", bucket)

    TRAIN_S3_URI = f"s3://{bucket}/data/Train"
    TEST_S3_URI = f"s3://{bucket}/data/Test"
    OUTPUT_S3_URI = f"s3://{bucket}/output"

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",  # assumes train.py is at repo root
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

    print("TRAIN_S3_URI =", TRAIN_S3_URI)
    print("TEST_S3_URI  =", TEST_S3_URI)
    print("OUTPUT_S3_URI =", OUTPUT_S3_URI)

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
    model_data = estimator.model_data
    print(model_data)
    print(f"Latest training job name: {estimator.latest_training_job.name}")
    print(f"Model artifact S3 URI: {estimator.model_data}")

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        entry_point = 'inference.py',
        source_dir = 'src',
        serializer=IdentitySerializer(content_type="image/jpeg"),
        deserializer = JSONDeserializer(),
    )

    results = predict_s3_folder(
        predictor,
        bucket=bucket,
        prefix="data/Test/"   
    )

    print("Total predictions:", len(results))

    predictor.delete_endpoint()

if __name__ == "__main__":
    main()

from sagemaker.pytorch import PyTorchModel
import sagemaker

model = PyTorchModel(
    model_data= "s3://sagemaker-us-east-1-327481844722/output/securecontent-cnn-train-2026-04-14-16-01-20-930/output/model.tar.gz",   # ← comes from training
    role=ROLE_ARN,
    entry_point="inference.py",
    source_dir="src",
    framework_version="2.4.0",
    py_version=PY_VERSION,
    sagemaker_session=sagemaker.Session(),
)

model_package = model.register(
    content_types=["image/jpeg", "image/png", "image/webp"],
    response_types=["application/json"],
    inference_instances=[INSTANCE_TYPE],
    transform_instances=[INSTANCE_TYPE],
    model_package_group_name="securecontent-model-group",  # ← name you choose
    approval_status="PendingManualApproval",  # or "Approved"
)

print("Model registered successfully!")
print(model_package)