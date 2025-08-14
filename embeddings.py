import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Tuple

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


class HistoricalEmbeddings:
    """
    SangamGPT Embeddings - Clean implementation for historical content similarity
    """
    
    def __init__(self):
        self.embedding_model = "models/text-embedding-004"
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for given text."""
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="semantic_similarity"
            )
            return response['embedding']
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        magnitude1 = np.linalg.norm(vec1_np)
        magnitude2 = np.linalg.norm(vec2_np)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_most_similar(self, query: str, documents: Dict[str, str]) -> Tuple[str, float]:
        """Find most similar document to query using embeddings."""
        print(f"🔍 Finding similar content for: '{query}'")
        print("=" * 60)
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            return "", 0.0
        
        best_match = ""
        highest_similarity = 0.0
        
        # Compare with each document
        for title, content in documents.items():
            doc_embedding = self.generate_embedding(content)
            if not doc_embedding:
                continue
                
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            
            print(f"📄 {title}: {similarity:.3f}")
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = title
        
        print(f"\n🎯 Best Match: {best_match} ({highest_similarity:.3f})")
        return best_match, highest_similarity
    
    def generate_contextual_response(self, query: str, context: str) -> str:
        """Generate AI response using retrieved context."""
        prompt = f"""
You are a historical expert. Use the provided context to answer the user's question accurately.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Base your answer primarily on the provided context
- If context is insufficient, clearly state so
- Provide specific historical details when available
- Keep response focused and informative
"""
        
        try:
            response = self.chat_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,  # Lower for accuracy
                    "top_k": 50,
                    "max_output_tokens": 400,
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"❌ Response Error: {e}"


def demonstrate_embeddings():
    """Demonstrate embeddings with historical content similarity."""
    print("🏛️ SangamGPT Historical Embeddings Demo")
    print("=" * 70)
    print("Showing how embeddings find semantically similar historical content\n")
    
    # Historical knowledge base
    historical_documents = {
        "Mauryan Empire": """
        The Mauryan Empire (322-185 BCE) was the first unified empire of ancient India, 
        founded by Chandragupta Maurya. It reached its peak under Ashoka the Great, who 
        ruled from 268-232 BCE. After the Kalinga War, Ashoka embraced Buddhism and 
        promoted non-violence throughout his vast empire, which stretched from Afghanistan 
        to southern India.
        """,
        
        "Mughal Architecture": """
        Mughal architecture flourished during the 16th-18th centuries, blending Persian, 
        Islamic, and Indian styles. The Taj Mahal, built by Shah Jahan as a mausoleum 
        for his wife Mumtaz Mahal, represents the pinnacle of this architectural style. 
        Other notable examples include the Red Fort, Humayun's Tomb, and Fatehpur Sikri.
        """,
        
        "Gupta Golden Age": """
        The Gupta Empire (320-550 CE) is considered the Golden Age of ancient India. 
        Under rulers like Chandragupta II, arts, science, and literature flourished. 
        Kalidasa wrote his masterpieces, mathematicians like Aryabhata made groundbreaking 
        discoveries, and the decimal system was developed. Universities like Nalanda 
        attracted scholars from across Asia.
        """,
        
        "Chola Maritime Power": """
        The Chola dynasty (9th-13th centuries) was a powerful South Indian empire known 
        for its naval prowess and overseas conquests. Rajendra Chola I conquered parts 
        of Southeast Asia, establishing trade networks across the Indian Ocean. The Cholas 
        built magnificent temples like Brihadeeswarar Temple and were patrons of Tamil 
        literature and bronze sculpture.
        """
    }
    
    embeddings = HistoricalEmbeddings()
    
    # Example query
    user_query = "Tell me about ancient Indian rulers who promoted Buddhism"
    
    # Find most relevant document
    best_match, similarity_score = embeddings.find_most_similar(
        user_query, 
        historical_documents
    )
    
    if best_match and similarity_score > 0.5:
        print(f"\n📚 Retrieved Context: {best_match}")
        context = historical_documents[best_match]
        
        print(f"\n🤖 Generating AI Response...")
        print("=" * 60)
        
        response = embeddings.generate_contextual_response(user_query, context)
        print(response)
        
    else:
        print("\n❌ No sufficiently similar content found")
    
  
    print(f"📊 Similarity threshold: 0.5 | Best match score: {similarity_score:.3f}")


if __name__ == "__main__":
    demonstrate_embeddings()
