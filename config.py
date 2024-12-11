import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Embedding Model Configuration
EMBEDDING_MODEL = 'models/embedding-001'

# Vector Store Configuration
VECTOR_STORE_PATH = './college_vectorstore'