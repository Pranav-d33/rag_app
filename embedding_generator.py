import google.generativeai as genai
from config import GEMINI_API_KEY, EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        # Configure the Gemini API
        genai.configure(api_key=GEMINI_API_KEY)

    def generate_embeddings(self, texts):
        """
        Generate embeddings for given texts using Gemini embedding model
        
        Args:
            texts (list): List of text strings to embed
        
        Returns:
            list: List of embeddings
        """
        embedding_model = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document"
        )
         # Debugging output
        return embedding_model.get('embeddings', [])


    def generate_query_embedding(self, query):
        """
        Generate embedding for a query
        
        Args:
            query (str): Query string to embed
        
        Returns:
            list: Embedding for the query
        """
        query_embedding = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        return query_embedding['embedding']