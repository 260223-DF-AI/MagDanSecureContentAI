from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

load_dotenv()
CS = os.getenv("CS")

def get_engine():
    if CS:
        engine = create_engine(CS)
        return engine
    else:
        raise ValueError("Connection String not found. Verify you .env file exists and includes this line: CS='your_postgresql_url")
