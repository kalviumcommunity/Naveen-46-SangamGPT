import os
import math
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Tuple, Dict

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


class CosineSimilarityAnalyzer:
    """
    Demonstrates cosine similarity with embeddings for historical text analysis.
    Clean implementation with one comprehensive example.
    """
    
    def __init__(self):
        self.model = "models/text-embedding-004"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Gemini."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="semantic_similarity"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        Where:
        - A · B is the dot product
        - ||A|| and ||B|| are the magnitudes (lengths) of vectors
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product (A · B)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes (||A|| and ||B||)
        magnitude_a = math.sqrt(sum(a * a for a in vec1))
        magnitude_b = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude_a * magnitude_b)
        return cosine_sim
    
    def analyze_historical_texts(self) -> Dict[str, any]:
        """
        Analyze similarity between different historical texts using cosine similarity.
        """
        print("🧮 SangamGPT Cosine Similarity Analysis")
        print("=" * 60)
        print("Analyzing historical text similarities using embeddings\n")
        
        # Historical texts about different topics
        historical_texts = {
            "Mauryan Empire": "The Mauryan Empire was founded by Chandragupta Maurya in 321 BCE. It became one of the largest empires in ancient India, spanning most of the Indian subcontinent. The empire reached its peak under Emperor Ashoka, who promoted Buddhism and non-violence after the Kalinga War.",
            
            "Mughal Empire": "The Mughal Empire was established in 1526 by Babur after his victory at the Battle of Panipat. The empire flourished under rulers like Akbar, who promoted religious tolerance and cultural synthesis. The Mughals were known for their architectural marvels like the Taj Mahal.",
            
            "Ancient Trade Routes": "The Silk Road was an extensive network of trade routes connecting East and West. Merchants traveled these dangerous paths carrying silk, spices, and precious goods. These routes facilitated cultural exchange and the spread of ideas, religions, and technologies.",
            
            "Ashoka's Philosophy": "Emperor Ashoka embraced Buddhism after witnessing the devastation of the Kalinga War. He promoted dharma, non-violence, and religious tolerance throughout his empire. His edicts, carved on rocks and pillars, spread Buddhist principles across ancient India."
        }
        
        # Generate embeddings for all texts
        print("📊 Generating embeddings for historical texts...")
        embeddings = {}
        for title, text in historical_texts.items():
            embedding = self.get_embedding(text)
            if embedding:
                embeddings[title] = embedding
                print(f"✓ Generated embedding for: {title} (dimension: {len(embedding)})")
        
        print(f"\n🎯 Calculating cosine similarities between all text pairs...\n")
        
        # Calculate all pairwise similarities
        similarities = []
        texts_list = list(embeddings.keys())
        
        for i, text1 in enumerate(texts_list):
            for j, text2 in enumerate(texts_list):
                if i < j:  # Only calculate each pair once
                    similarity = self.cosine_similarity(embeddings[text1], embeddings[text2])
                    similarities.append((text1, text2, similarity))
                    
                    # Interpret similarity level
                    if similarity > 0.8:
                        level = "Very High"
                    elif similarity > 0.6:
                        level = "High"
                    elif similarity > 0.4:
                        level = "Moderate"
                    elif similarity > 0.2:
                        level = "Low"
                    else:
                        level = "Very Low"
                    
                    print(f"📈 {text1} ↔ {text2}")
                    print(f"   Similarity: {similarity:.4f} ({level})")
                    print()
        
        # Find most and least similar pairs
        similarities.sort(key=lambda x: x[2], reverse=True)
        most_similar = similarities[0]
        least_similar = similarities[-1]
        
        print("🏆 ANALYSIS RESULTS:")
        print("=" * 50)
        print(f"Most Similar Texts:")
        print(f"   {most_similar[0]} ↔ {most_similar[1]}")
        print(f"   Cosine Similarity: {most_similar[2]:.4f}")
        print()
        print(f"Least Similar Texts:")
        print(f"   {least_similar[0]} ↔ {least_similar[1]}")
        print(f"   Cosine Similarity: {least_similar[2]:.4f}")
        print()
        
        return {
            'embeddings': embeddings,
            'similarities': similarities,
            'most_similar': most_similar,
            'least_similar': least_similar
        }
    
    def explain_cosine_similarity(self):
        """Explain how cosine similarity works with embeddings."""
        print("📚 UNDERSTANDING COSINE SIMILARITY")
        print("=" * 60)
        print()
        print("🔍 What is Cosine Similarity?")
        print("   Measures the cosine of the angle between two vectors")
        print("   Values range from -1 to 1:")
        print("   • 1.0 = Identical direction (most similar)")
        print("   • 0.0 = Perpendicular (no similarity)")
        print("   • -1.0 = Opposite direction (least similar)")
        print()
        print("📐 Mathematical Formula:")
        print("   cos(θ) = (A · B) / (||A|| × ||B||)")
        print("   Where:")
        print("   • A · B = dot product of vectors")
        print("   • ||A|| = magnitude of vector A")
        print("   • ||B|| = magnitude of vector B")
        print()
        print("🎯 Why Use Cosine Similarity for Text?")
        print("   • Measures semantic similarity, not just word overlap")
        print("   • Handles different text lengths effectively")
        print("   • Focuses on direction, not magnitude")
        print("   • Perfect for high-dimensional embeddings")
        print()
        print("🧠 In LLMs and Embeddings:")
        print("   • Embeddings capture semantic meaning in vector space")
        print("   • Similar meanings → vectors point in similar directions")
        print("   • Cosine similarity measures this directional similarity")
        print("   • Essential for RAG, search, and recommendation systems")
        print()


def demonstrate_cosine_similarity():
    """
    Main demonstration of cosine similarity with historical embeddings.
    """
    analyzer = CosineSimilarityAnalyzer()
    
    # Explain the concept first
    analyzer.explain_cosine_similarity()
    
    # Run the analysis
    results = analyzer.analyze_historical_texts()
    
    print("✅ COSINE SIMILARITY DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("Key Insights:")
    print("• Texts about the same empire/ruler show higher similarity")
    print("• Different topics (trade vs. philosophy) show lower similarity")  
    print("• Cosine similarity captures semantic relationships effectively")
    print("• This technique is fundamental for AI search and retrieval systems")
    
    return results


if __name__ == "__main__":
    demonstrate_cosine_similarity()
