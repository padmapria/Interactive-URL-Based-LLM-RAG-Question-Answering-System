# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Accessing environment variables using os.getenv
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    EMAIL = os.getenv('EMAIL')
    TOOL_NAME = os.getenv('TOOL_NAME')
    MAX_RESULTS = os.getenv('MAX_RESULTS', 1)  # Providing a default value if not set

