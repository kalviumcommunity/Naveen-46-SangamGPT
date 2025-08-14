import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List

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


class TopKExplorer:
    """
    Demonstrates how different Top K settings affect AI responses
    in historical research contexts.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens = 0
    
    def log_tokens(self, prompt: str, response: str):
        """Log estimated token usage."""
        tokens = len((prompt + response).split()) * 0.75
        self.total_tokens += tokens
        print(f"📊 Tokens: {int(tokens)} | Total: {int(self.total_tokens)}")
    
    def generate_with_top_k(self, prompt: str, top_k: int, purpose: str, temperature: float = 0.7) -> str:
        """Generate content with a specific Top K setting."""
        print(f"🔢 Top K: {top_k} | 🌡️ Temp: {temperature} | {purpose}")
        print("=" * 50)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": 0.9,
                    "max_output_tokens": 300,
                }
            )
            result = response.text.strip()
            self.log_tokens(prompt, result)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def historical_fact_test(self, top_k: int) -> str:
        """Test Top K with factual historical questions."""
        prompt = "Who was Chandragupta Maurya and what were his major achievements? Provide key dates."
        return self.generate_with_top_k(prompt, top_k, "Historical Facts", temperature=0.3)
    
    def creative_story_test(self, top_k: int) -> str:
        """Test Top K with creative historical storytelling."""
        prompt = "Describe attending Emperor Akbar's court session with vivid details and atmosphere."
        return self.generate_with_top_k(prompt, top_k, "Creative Story", temperature=0.8)


def demonstrate_top_k_effects():
    """Demonstrate Top K effects with two clean examples."""
    print("🔢 SangamGPT Top K Demonstration")
    print("=" * 60)
    
    explorer = TopKExplorer()
    top_k_values = [20, 50, 150]
    
    # Example 1: Historical Facts
    print("\n📚 EXAMPLE 1: HISTORICAL FACTS")
    print("=" * 40)
    for top_k in top_k_values:
        result = explorer.historical_fact_test(top_k)
        print(f"Result: {result}\n")
    
    # Example 2: Creative Storytelling  
    print("\n🎨 EXAMPLE 2: CREATIVE STORYTELLING")
    print("=" * 40)
    for top_k in top_k_values:
        result = explorer.creative_story_test(top_k)
        print(f"Result: {result}\n")


class TopKComparison:
    """Compare the same prompt with different Top K values."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def compare_side_by_side(self, prompt: str, top_k_values: List[int]) -> Dict[int, str]:
        """Run same prompt with different Top K values."""
        results = {}
        print(f"🔄 Side-by-Side Top K Comparison")
        print("=" * 50)
        
        for top_k in top_k_values:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_k": top_k,
                        "top_p": 0.9,
                        "max_output_tokens": 200,
                    }
                )
                results[top_k] = response.text.strip()
                print(f"🔢 Top K {top_k}: {results[top_k]}")
                print("-" * 40)
            except Exception as e:
                results[top_k] = f"Error: {str(e)}"
        return results


def run_comparison():
    """Demonstrate same prompt with different Top K values."""
    print("\n📊 TOP K COMPARISON")
    print("=" * 50)
    
    comparator = TopKComparison()
    prompt = "Explain the significance of the Battle of Panipat (1526)."
    top_k_values = [30, 80, 150]
    
    results = comparator.compare_side_by_side(prompt, top_k_values)
    
    print("\n📋 Analysis:")
    print("• Lower Top K (30): Predictable, focused vocabulary")
    print("• Medium Top K (80): Balanced variety and coherence")
    print("• Higher Top K (150): Diverse, creative expressions")


def show_recommendations():
    """Show Top K recommendations for different use cases."""
    print("\n🎯 TOP K RECOMMENDATIONS")
    print("=" * 50)
    print("📖 Historical Facts: Top K 20-40 (Conservative)")
    print("📚 Educational Content: Top K 40-80 (Balanced)")  
    print("🎨 Creative Writing: Top K 80-200 (Diverse)")
    print("💭 Brainstorming: Top K 100-300 (Maximum variety)")


if __name__ == "__main__":
    demonstrate_top_k_effects()
    run_comparison()
    show_recommendations()
    
    print("\n✅ Top K demonstration complete!")
    print("Top K controls vocabulary size for precise AI creativity control.")
