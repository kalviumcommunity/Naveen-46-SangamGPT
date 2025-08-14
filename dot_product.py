import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Tuple
import math

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"❌ API Configuration Error: {e}")
    exit(1)


class DotProductSimilarity:
    """
    Implements dot product similarity for comparing historical texts using embeddings.
    Demonstrates how semantic similarity works in SangamGPT.
    """
    
    def __init__(self):
        self.total_tokens = 0
    
    def log_tokens(self, text: str, operation: str):
        """Log estimated token usage."""
        tokens = len(text.split()) * 1.3  # Embeddings use more tokens
        self.total_tokens += tokens
        print(f"📊 {operation} Tokens: {int(tokens)} | Total: {int(self.total_tokens)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text using Gemini.
        """
        try:
            self.log_tokens(text, "Embedding")
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def dot_product_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate dot product similarity between two embeddings.
        
        Dot Product = sum(a[i] * b[i] for all i)
        Higher values indicate greater similarity.
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        return dot_product
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity for comparison with dot product.
        
        Cosine Similarity = dot_product / (magnitude1 * magnitude2)
        Range: -1 to 1, where 1 is identical, 0 is orthogonal, -1 is opposite.
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def compare_historical_texts(self, texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple historical texts using dot product similarity.
        """
        print("🔍 Generating embeddings for historical texts...")
        print("=" * 60)
        
        # Generate embeddings for all texts
        embeddings = {}
        for name, text in texts.items():
            print(f"📚 Processing: {name}")
            embeddings[name] = self.generate_embedding(text)
        
        print("\n🧮 Computing similarities...")
        print("=" * 60)
        
        # Calculate similarities
        similarities = {}
        text_names = list(texts.keys())
        
        for i, text1 in enumerate(text_names):
            similarities[text1] = {}
            for j, text2 in enumerate(text_names):
                if i != j:  # Don't compare text with itself
                    dot_product = self.dot_product_similarity(embeddings[text1], embeddings[text2])
                    cosine_sim = self.cosine_similarity(embeddings[text1], embeddings[text2])
                    similarities[text1][text2] = {
                        'dot_product': dot_product,
                        'cosine_similarity': cosine_sim
                    }
        
        return similarities
    
    def find_most_similar(self, query_text: str, document_texts: Dict[str, str]) -> Tuple[str, float, float]:
        """
        Find the most similar document to a query using dot product.
        Returns: (most_similar_name, dot_product_score, cosine_score)
        """
        print(f"🔎 Finding most similar text to query...")
        print("=" * 50)
        
        # Generate embedding for query
        query_embedding = self.generate_embedding(query_text)
        
        # Generate embeddings for documents
        doc_embeddings = {}
        for name, text in document_texts.items():
            doc_embeddings[name] = self.generate_embedding(text)
        
        # Find most similar
        best_match = None
        best_dot_product = float('-inf')
        best_cosine = 0
        
        for name, doc_embedding in doc_embeddings.items():
            dot_product = self.dot_product_similarity(query_embedding, doc_embedding)
            cosine_sim = self.cosine_similarity(query_embedding, doc_embedding)
            
            print(f"📖 {name}:")
            print(f"   Dot Product: {dot_product:.4f}")
            print(f"   Cosine Similarity: {cosine_sim:.4f}")
            print()
            
            if dot_product > best_dot_product:
                best_dot_product = dot_product
                best_cosine = cosine_sim
                best_match = name
        
        return best_match, best_dot_product, best_cosine


def demonstrate_dot_product_similarity():
    """
    Demonstrate dot product similarity with historical texts from Indian history.
    """
    print("🎯 SangamGPT Dot Product Similarity Demonstration")
    print("=" * 70)
    print("Comparing historical texts using embeddings and dot product similarity")
    print()
    
    similarity_analyzer = DotProductSimilarity()
    
    # Historical texts about different Indian empires
    historical_texts = {
        "Mauryan Empire": """
        The Mauryan Empire (322-185 BCE) was one of the largest empires in Indian history. 
        Founded by Chandragupta Maurya, it reached its peak under Ashoka the Great. 
        The empire was known for its sophisticated administrative system, extensive trade networks, 
        and Ashoka's promotion of Buddhism after the Kalinga War. The capital was Pataliputra, 
        and the empire controlled most of the Indian subcontinent.
        """,
        
        "Mughal Empire": """
        The Mughal Empire (1526-1857) was established by Babur and reached its zenith under Akbar. 
        Known for magnificent architecture like the Taj Mahal, the Mughals created a centralized 
        administrative system and promoted cultural synthesis. The empire was famous for its 
        military prowess, economic prosperity, and artistic achievements. Delhi and Agra served 
        as major capitals during different periods.
        """,
        
        "Gupta Empire": """
        The Gupta Empire (320-550 CE) is often called the Golden Age of India. 
        Under rulers like Chandragupta II, it witnessed remarkable progress in science, 
        mathematics, astronomy, literature, and arts. Kalidasa flourished during this period. 
        The empire was known for religious tolerance, economic prosperity, and cultural achievements. 
        Universities like Nalanda attracted scholars from across Asia.
        """
    }
    
    # Compare all texts with each other
    similarities = similarity_analyzer.compare_historical_texts(historical_texts)
    
    # Display results in a clean format
    print("\n📊 SIMILARITY RESULTS")
    print("=" * 70)
    
    for text1, comparisons in similarities.items():
        print(f"\n📚 {text1} compared to:")
        for text2, scores in comparisons.items():
            dot_product = scores['dot_product']
            cosine_sim = scores['cosine_similarity']
            print(f"   📖 {text2}:")
            print(f"      • Dot Product: {dot_product:.4f}")
            print(f"      • Cosine Similarity: {cosine_sim:.4f}")
        print("-" * 50)
    
    # Demonstrate query-based similarity
    print("\n🔍 QUERY-BASED SIMILARITY SEARCH")
    print("=" * 70)
    
    query = "Ancient Indian empire known for Buddhist influence and administrative excellence"
    
    best_match, dot_score, cosine_score = similarity_analyzer.find_most_similar(query, historical_texts)
    
    print(f"📝 Query: '{query}'")
    print(f"🏆 Best Match: {best_match}")
    print(f"📊 Dot Product Score: {dot_score:.4f}")
    print(f"📊 Cosine Similarity: {cosine_score:.4f}")
    
    # Explain the results
    print("\n💡 INTERPRETATION")
    print("=" * 50)
    print("🔹 Dot Product:")
    print("   - Higher values = more similar")
    print("   - Considers both angle and magnitude of vectors")
    print("   - Good for finding semantically related content")
    print()
    print("🔹 Cosine Similarity:")
    print("   - Range: -1 to 1 (1 = identical, 0 = unrelated)")
    print("   - Measures angle between vectors (ignores magnitude)")
    print("   - Better for normalized comparisons")
    print()
    print("🔹 For SangamGPT:")
    print("   - Use dot product for finding relevant historical content")
    print("   - Use cosine similarity for ranking document relevance")
    print("   - Both help build intelligent historical search systems")


if __name__ == "__main__":
    demonstrate_dot_product_similarity()
    print("\n✅ Dot product similarity demonstration complete!")
    print("This shows how embeddings enable semantic search in historical texts.")
