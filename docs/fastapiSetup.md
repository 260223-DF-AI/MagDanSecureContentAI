# Steps to set up the FastAPI service
1. run "deploy_registry_model" to get the endpoint name
2. take that endpoint name, set it into your environment variables
- will also have to input that endpoint name into core.settings as "vision_endpoint"
3. run main.py using uvicorn in your terminal
- uvicorn src.main:app --reload
4. go to your browser, input the "link" it gives you in the terminal
- add /docs#/ to see the get, post requests