#!/usr/bin/env bash
# =============================================================================
# ResearchFlow — Deployment Script
# =============================================================================
# Builds and deploys the Lambda function using AWS SAM.
#
# Prerequisites:
#   - AWS CLI configured with valid credentials
#   - AWS SAM CLI installed (pip install aws-sam-cli)
#   - Docker installed (for sam build)
#
# Usage:
#   cd deployment && bash deploy.sh
# =============================================================================

set -euo pipefail

echo "Building SAM application..."
sam build --template-file template.yaml

echo "Deploying to AWS..."
sam deploy \
  --guided \
  --stack-name researchflow \
  --capabilities CAPABILITY_IAM

echo "Deployment complete. Check the Outputs above for your API endpoint."
