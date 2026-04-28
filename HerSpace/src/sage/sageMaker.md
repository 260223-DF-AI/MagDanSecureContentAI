* NOTE: all this information in this folder "sage", is what is happening in SageMaker.  This is just to keep note of it in our repo, so we can reference it.

### Run Order:
1. Upload your dataset folders to S3.
- change artifacts to the actual thing as well
2. Run python launch_training.py
3. Copy the printed Model artifact S3 URI
4. Set that as MODEL_ARTIFACT_S3_URI
5. Run python register_model.py
- this (the updated version) is at the bottom of "launch_training.py"