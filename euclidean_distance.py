import os
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Tuple
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


class EuclideanDistanceSimilarity:
    """
    Implements Euclidean Distance (L2 Distance) for measuring similarity 
    between text embeddings in historical research contexts.
    """
    
    def __init__(self):
        self.total_tokens = 0
    
    def log_tokens(self, text: str):
        """Log estimated token usage."""
        tokens = len(text.split()) * 0.75
        self.total_tokens += tokens
        print(f"📊 Tokens: {int(tokens)} | Total: {int(self.total_tokens)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Gemini API."""
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            self.log_tokens(text)
            return np.array(response['embedding'])
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return np.zeros(768)
    
    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean Distance (L2 Distance) between two vectors.
        
        Formula: distance = sqrt(sum((vec1[i] - vec2[i])^2))
        Lower distance = higher similarity
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        squared_differences = (vec1 - vec2) ** 2
        distance = math.sqrt(np.sum(squared_differences))
        return distance
    
    def euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Convert Euclidean distance to similarity score.
        Uses: similarity = 1 / (1 + distance)
        Range: 0 to 1 (1 = identical, closer to 0 = more different)
        """
        distance = self.euclidean_distance(vec1, vec2)
        similarity = 1 / (1 + distance)
        return similarity
    
    def compare_historical_texts(self, texts: List[str], labels: List[str]) -> List[Tuple[str, str, float, float]]:
        """
        Compare multiple historical texts using Euclidean distance.
        Returns: [(text1, text2, distance, similarity), ...]
        """
        print("🔄 Generating embeddings for historical texts...")
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"📝 Processing: {labels[i]}")
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        print("\n📏 Calculating Euclidean distances...")
        results = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                distance = self.euclidean_distance(embeddings[i], embeddings[j])
                similarity = self.euclidean_similarity(embeddings[i], embeddings[j])
                results.append((labels[i], labels[j], distance, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[3], reverse=True)
        return results
    
    def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                         candidate_labels: List[str], top_k: int = 3) -> List[Tuple[str, float, float]]:
        """
        Find most similar texts to a query using Euclidean distance.
        Returns: [(label, distance, similarity), ...]
        """
        print(f"🔍 Query: {query_text[:60]}...")
        query_embedding = self.get_embedding(query_text)
        
        results = []
        for i, text in enumerate(candidate_texts):
            candidate_embedding = self.get_embedding(text)
            distance = self.euclidean_distance(query_embedding, candidate_embedding)
            similarity = self.euclidean_similarity(query_embedding, candidate_embedding)
            results.append((candidate_labels[i], distance, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


def demonstrate_euclidean_distance():
    """
    Demonstrate Euclidean distance similarity with historical texts.
    """
    print("📏 SangamGPT Euclidean Distance Similarity Demo")
    print("=" * 70)
    
    euclidean_calc = EuclideanDistanceSimilarity()
    
    # Historical texts about different empires
    historical_texts = [
        "The Mauryan Empire was founded by Chandragupta Maurya and reached its peak under Ashoka the Great, who promoted Buddhism and non-violence.",
        "The Gupta Empire is known as the Golden Age of India, with remarkable achievements in mathematics, astronomy, literature, and arts under rulers like Chandragupta II.",
        "The Mughal Empire was established by Babur and became famous for architectural marvels like the Taj Mahal, built during Shah Jahan's reign.",
        "The Chola Empire was a powerful South Indian dynasty known for maritime trade, naval expeditions, and magnificent temple architecture."
    ]
    
    text_labels = [
        "Mauryan Empire",
        "Gupta Empire", 
        "Mughal Empire",
        "Chola Empire"
    ]
    
    # Compare all texts with each other
    print("\n🔍 PAIRWISE COMPARISON USING EUCLIDEAN DISTANCE")
    print("=" * 70)
    
    comparison_results = euclidean_calc.compare_historical_texts(historical_texts, text_labels)
    
    print("\n📊 Similarity Results (Ranked by Euclidean Similarity):")
    print("-" * 60)
    
    for i, (text1, text2, distance, similarity) in enumerate(comparison_results, 1):
        print(f"{i}. {text1} ↔ {text2}")
        print(f"   📏 Euclidean Distance: {distance:.4f}")
        print(f"   🎯 Similarity Score: {similarity:.4f}")
        print(f"   💭 Interpretation: {'Very Similar' if similarity > 0.8 else 'Moderately Similar' if similarity > 0.6 else 'Different'}")
        print("-" * 40)
    
    # Query-based search
    print("\n🔍 QUERY-BASED SIMILARITY SEARCH")
    print("=" * 70)
    
    queries = [
        "Which empire was known for promoting Buddhism and peace?",
        "Tell me about architectural achievements and monuments"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        search_results = euclidean_calc.find_most_similar(
            query, historical_texts, text_labels, top_k=2
        )
        
        for rank, (label, distance, similarity) in enumerate(search_results, 1):
            print(f"🏆 Rank {rank}: {label}")
            print(f"   📏 Distance: {distance:.4f}")
            print(f"   🎯 Similarity: {similarity:.4f}")
            print("-" * 30)
    
    # Educational comparison with other similarity methods
    print("\n📚 EUCLIDEAN DISTANCE vs OTHER SIMILARITY METHODS")
    print("=" * 70)
    print("🔍 Understanding the differences:")
    print()
    print("📏 EUCLIDEAN DISTANCE:")
    print("   • Measures actual 'distance' between vectors in space")
    print("   • Sensitive to vector magnitude (length)")
    print("   • Range: 0 to ∞ (0 = identical)")
    print("   • Good for: Clustering, nearest neighbor search")
    print()
    print("📐 COSINE SIMILARITY (Alternative):")
    print("   • Measures angle between vectors")
    print("   • Ignores vector magnitude, focuses on direction")
    print("   • Range: -1 to 1 (1 = identical)")
    print("   • Good for: Text similarity, recommendation systems")
    print()
    print("🔢 DOT PRODUCT (Alternative):")
    print("   • Simple multiplication and sum")
    print("   • Sensitive to both angle and magnitude")
    print("   • Range: -∞ to ∞")
    print("   • Good for: When magnitude matters")
    
    return euclidean_calc


def demonstrate_distance_properties():
    """
    Demonstrate mathematical properties of Euclidean distance.
    """
    print("\n🧮 EUCLIDEAN DISTANCE MATHEMATICAL PROPERTIES")
    print("=" * 70)
    
    # Create simple example vectors
    vec_a = np.array([1, 2, 3])
    vec_b = np.array([4, 5, 6])
    vec_c = np.array([1, 2, 3])  # Same as vec_a
    
    calc = EuclideanDistanceSimilarity()
    
    print("📊 Example Vectors:")
    print(f"   Vector A: {vec_a}")
    print(f"   Vector B: {vec_b}")
    print(f"   Vector C: {vec_c} (identical to A)")
    print()
    
    # Property 1: Identity
    dist_aa = calc.euclidean_distance(vec_a, vec_a)
    print(f"🔍 Property 1 - Identity: distance(A,A) = {dist_aa:.4f}")
    print("   ✓ Distance from a vector to itself is always 0")
    print()
    
    # Property 2: Symmetry
    dist_ab = calc.euclidean_distance(vec_a, vec_b)
    dist_ba = calc.euclidean_distance(vec_b, vec_a)
    print(f"🔍 Property 2 - Symmetry: distance(A,B) = {dist_ab:.4f}")
    print(f"                           distance(B,A) = {dist_ba:.4f}")
    print("   ✓ Distance is symmetric")
    print()
    
    # Property 3: Identical vectors
    dist_ac = calc.euclidean_distance(vec_a, vec_c)
    sim_ac = calc.euclidean_similarity(vec_a, vec_c)
    print(f"🔍 Property 3 - Identical: distance(A,C) = {dist_ac:.4f}")
    print(f"                           similarity(A,C) = {sim_ac:.4f}")
    print("   ✓ Identical vectors have distance 0 and similarity 1")


if __name__ == "__main__":
    # Run the comprehensive Euclidean distance demonstration
    calculator = demonstrate_euclidean_distance()
    
    # Demonstrate mathematical properties
    demonstrate_distance_properties()
    
    print(f"\n📈 Session Statistics:")
    print(f"🪙 Total Tokens Used: {int(calculator.total_tokens)}")
    
    print("\n✅ Euclidean Distance Demo Complete!")
    print("This demonstrates L2 distance for measuring text embedding similarity.")
    print("Euclidean distance is fundamental for clustering and nearest neighbor search!")
