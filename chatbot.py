import google.generativeai as genai
from config import GEMINI_API_KEY
from retriever import Retriever

class CollegeChatbot:
    def __init__(self, knowledge_base):
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Initialize retriever
        self.retriever = Retriever(knowledge_base)
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, query):
        """
        Generate a response based on retrieved context and query
        
        Args:
            query (str): User's query
        
        Returns:
            str: AI-generated response
        """
        # Retrieve relevant college information
        retrieved_docs = self.retriever.retrieve(query)
        
        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"College: {doc['name']}\n"
            f"Location: {doc['location']}\n"
            f"Courses: {', '.join(doc['courses'])}\n"
            f"Placement Records: {doc['placement_records']}\n"
            f"Average Package: {doc['average_package']}"
            for doc in retrieved_docs
        ])
        
        # Construct prompt with context
        full_prompt = f"""
        Context Information:
        {context}
        
        User Query: {query}
        
        Based on the context and query, provide a helpful and informative response about the colleges.
        """
        
        # Generate response using Gemini
        response = self.model.generate_content(full_prompt)
        return response.text