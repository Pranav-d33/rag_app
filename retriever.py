from vector_store import VectorStore

class Retriever:
    def __init__(self, knowledge_base):
        self.vector_store = VectorStore(knowledge_base)
    
    def retrieve(self, query, top_k=3):
        """
        Retrieve top k most relevant documents for a query
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to retrieve
        
        Returns:
            list: Top matching college entries
        """
        return self.vector_store.similarity_search(query, top_k)