"""
ResearchFlow — AWS Lambda Handler

Entry point for the serverless deployment behind API Gateway.
Receives a POST /research request and invokes the Supervisor graph.
"""

import json


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda handler for the /research endpoint.

    TODO:
    - Parse the request body from event.
    - Extract the "question" field.
    - Initialize and invoke the Supervisor graph.
    - Return the structured research report as JSON.
    - Handle errors gracefully with appropriate HTTP status codes.

    Expected request:  { "question": "..." }
    Expected response: { "statusCode": 200, "body": "<JSON report>" }
    """
    raise NotImplementedError
