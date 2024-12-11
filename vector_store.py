import os
import numpy as np
from embedding_generator import EmbeddingGenerator
from config import VECTOR_STORE_PATH
import json

class VectorStore:
    def __init__(self, knowledge_base):
        self.embedding_generator = EmbeddingGenerator()
        self.knowledge_base = knowledge_base
        self.vector_store_path = VECTOR_STORE_PATH
        
        # Ensure vector store directory exists
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Prepare and store embeddings
        self.prepare_vector_store()

    def prepare_vector_store(self):
        """
        Prepare vector store by generating embeddings for college data
        """
        # Extract text for embedding
        texts = []
        text_metadata = []
        
        for college in self.knowledge_base:
            # Create comprehensive text representation
            college_text = f"""
            College Name: {college['name']}
            Location: {college['location']}
            Type: {college['type']}
            Established: {college['established']}
            Courses: {', '.join(college['courses'])}
            Placement Records: {college['placement_records']}
            Average Package: {college['average_package']}
            Highest Package: {college['highest_package']}
            Website: {college['website']}
            """
            texts.append(college_text)
            text_metadata.append(college)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Store embeddings and metadata
        for i, (embedding, metadata) in enumerate(zip(embeddings, text_metadata)):
            filename = os.path.join(self.vector_store_path, f'embedding_{i}.json')
            with open(filename, 'w') as f:
                json.dump({
                    'embedding': embedding,
                    'metadata': metadata
                }, f)

    def similarity_search(self, query, top_k=3):
        """
        Perform similarity search for a given query
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
        
        Returns:
            list: Top matching college entries
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Load and compare embeddings
        similarities = []
        metadata_results = []
        
        for filename in os.listdir(self.vector_store_path):
            if filename.startswith('embedding_'):
                filepath = os.path.join(self.vector_store_path, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, data['embedding']) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(data['embedding'])
                )
                similarities.append(similarity)
                metadata_results.append(data['metadata'])
        
        # Sort and return top k results
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return [metadata_results[i] for i in top_indices]