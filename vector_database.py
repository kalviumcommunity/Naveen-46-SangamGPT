import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
from datetime import datetime

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


class HistoricalVectorDatabase:
    """
    A simple vector database for storing and searching historical text embeddings.
    Uses in-memory storage with file persistence.
    """
    
    def __init__(self, db_file: str = "historical_vector_db.json"):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.db_file = db_file
        self.vectors: Dict[str, Dict] = {}
        self.load_database()
    
    def load_database(self):
        """Load existing vector database from file."""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert list back to numpy arrays
                    for doc_id, doc_data in data.items():
                        doc_data['embedding'] = np.array(doc_data['embedding'])
                    self.vectors = data
                print(f"📚 Loaded {len(self.vectors)} documents from database")
            else:
                print("📚 Starting with empty vector database")
        except Exception as e:
            print(f"⚠️ Error loading database: {e}")
            self.vectors = {}
    
    def save_database(self):
        """Save vector database to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {}
            for doc_id, doc_data in self.vectors.items():
                data_copy = doc_data.copy()
                data_copy['embedding'] = doc_data['embedding'].tolist()
                data_to_save[doc_id] = data_copy
            
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            print(f"💾 Database saved with {len(self.vectors)} documents")
        except Exception as e:
            print(f"⚠️ Error saving database: {e}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Gemini."""
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(response['embedding'])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(768)  # Fallback dimension
    
    def add_document(self, doc_id: str, text: str, metadata: Dict = None) -> bool:
        """Add a document to the vector database."""
        print(f"🔄 Processing document: {doc_id}")
        
        embedding = self.get_embedding(text)
        if embedding.size == 0:
            print(f"❌ Failed to generate embedding for {doc_id}")
            return False
        
        self.vectors[doc_id] = {
            'text': text,
            'embedding': embedding,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Added document: {doc_id} (embedding dim: {embedding.shape[0]})")
        return True
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norms == 0:
            return 0.0
        return dot_product / norms
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Search for most similar documents to the query."""
        if not self.vectors:
            print("📭 Database is empty")
            return []
        
        print(f"🔍 Searching for: '{query[:50]}...'")
        
        query_embedding = self.get_embedding(query)
        if query_embedding.size == 0:
            print("❌ Failed to generate query embedding")
            return []
        
        results = []
        for doc_id, doc_data in self.vectors.items():
            similarity = self.cosine_similarity(query_embedding, doc_data['embedding'])
            results.append((doc_id, similarity, doc_data['text']))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve a document by ID."""
        return self.vectors.get(doc_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the database."""
        return list(self.vectors.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the database."""
        if doc_id in self.vectors:
            del self.vectors[doc_id]
            print(f"🗑️ Deleted document: {doc_id}")
            return True
        return False


def demonstrate_vector_database():
    """Demonstrate vector database functionality with historical texts."""
    print("🏛️ SangamGPT Vector Database Demo")
    print("=" * 60)
    
    # Initialize vector database
    db = HistoricalVectorDatabase()
    
    # Historical documents to add
    historical_docs = {
        "mauryan_empire": {
            "text": "The Mauryan Empire (321-185 BCE) was the first pan-Indian empire, founded by Chandragupta Maurya. Under Ashoka the Great, it reached its zenith, covering most of the Indian subcontinent. The empire was known for its sophisticated administration, extensive road networks, and promotion of Buddhism.",
            "metadata": {"period": "Ancient India", "dynasty": "Mauryan", "ruler": "Chandragupta & Ashoka"}
        },
        "mughal_empire": {
            "text": "The Mughal Empire (1526-1857) was founded by Babur and reached its peak under Akbar, Jahangir, and Shah Jahan. Known for architectural marvels like the Taj Mahal, the empire blended Persian, Islamic, and Indian cultures. It established a centralized administration and promoted religious tolerance under Akbar.",
            "metadata": {"period": "Medieval India", "dynasty": "Mughal", "ruler": "Babur to Aurangzeb"}
        },
        "gupta_empire": {
            "text": "The Gupta Empire (320-550 CE) is considered the Golden Age of India. Under rulers like Chandragupta II, it saw unprecedented growth in arts, science, literature, and mathematics. Kalidasa flourished during this period, and the concept of zero was developed. Sanskrit literature and Hindu philosophy reached new heights.",
            "metadata": {"period": "Classical India", "dynasty": "Gupta", "ruler": "Chandragupta II"}
        },
        "chola_empire": {
            "text": "The Chola Empire (300-1279 CE) was a powerful South Indian dynasty known for maritime trade and naval expeditions. Under Rajaraja Chola and Rajendra Chola, they built magnificent temples like Brihadeeswarar Temple. The Cholas had extensive trade networks reaching Southeast Asia and were great patrons of Tamil literature.",
            "metadata": {"period": "Medieval India", "dynasty": "Chola", "ruler": "Rajaraja & Rajendra"}
        }
    }
    
    # Add documents to vector database
    print("\n📝 Adding Historical Documents...")
    print("-" * 40)
    for doc_id, doc_info in historical_docs.items():
        success = db.add_document(doc_id, doc_info["text"], doc_info["metadata"])
        if not success:
            print(f"❌ Failed to add {doc_id}")
    
    # Save database
    db.save_database()
    
    # Demonstrate similarity search
    print("\n🔍 Testing Similarity Search...")
    print("-" * 40)
    
    queries = [
        "Which empire was known for promoting Buddhism?",
        "Tell me about architectural achievements in medieval India",
        "Which dynasty was famous for maritime trade?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📊 Query {i}: {query}")
        print("=" * 50)
        
        results = db.search_similar(query, top_k=2)
        
        for rank, (doc_id, similarity, text) in enumerate(results, 1):
            doc_info = db.get_document(doc_id)
            metadata = doc_info['metadata'] if doc_info else {}
            
            print(f"🏆 Rank {rank} - {doc_id.replace('_', ' ').title()}")
            print(f"📊 Similarity: {similarity:.4f}")
            print(f"👑 Dynasty: {metadata.get('dynasty', 'Unknown')}")
            print(f"📜 Text: {text[:100]}...")
            print("-" * 30)
    
    # Database statistics
    print(f"\n📈 Database Statistics:")
    print("-" * 30)
    print(f"📚 Total Documents: {len(db.vectors)}")
    print(f"📂 Document IDs: {', '.join(db.list_documents())}")
    
    # Show embedding dimensions
    if db.vectors:
        sample_doc = next(iter(db.vectors.values()))
        print(f"🔢 Embedding Dimensions: {sample_doc['embedding'].shape[0]}")
    
    return db


def demonstrate_advanced_search(db: HistoricalVectorDatabase):
    """Demonstrate advanced search capabilities."""
    print("\n🎯 Advanced Vector Search Demo")
    print("=" * 60)
    
    # Complex queries
    complex_queries = [
        "Find empires that made significant contributions to art and culture",
        "Which rulers were known for religious tolerance and diversity?",
        "Show me information about ancient Indian mathematics and science"
    ]
    
    for query in complex_queries:
        print(f"\n🔎 Advanced Query: {query}")
        print("=" * 50)
        
        results = db.search_similar(query, top_k=2)
        
        for doc_id, similarity, text in results:
            print(f"📋 {doc_id.replace('_', ' ').title()}")
            print(f"📊 Relevance: {similarity:.4f}")
            print(f"💭 Excerpt: {text[:120]}...")
            print("-" * 25)


if __name__ == "__main__":
    # Run the comprehensive vector database demonstration
    vector_db = demonstrate_vector_database()
    
    # Run advanced search demo
    demonstrate_advanced_search(vector_db)
    
    print("This demonstrates how vector databases enable semantic search in historical research.")
    print("The database persists data between runs for continued learning and discovery.")
