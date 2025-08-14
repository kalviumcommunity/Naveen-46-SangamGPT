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


class TopPExplorer:
    """
    Demonstrates how different Top P settings affect AI responses
    in historical research contexts.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens = 0
    
    def log_tokens(self, prompt: str, response: str):
        """Log estimated token usage."""
        tokens = len((prompt + response).split()) * 0.75
        self.total_tokens += tokens
        print(f"📊 Tokens: {int(tokens)} | Session Total: {int(self.total_tokens)}")
        print("-" * 50)
    
    def generate_with_top_p(self, prompt: str, top_p: float, purpose: str, temperature: float = 0.7) -> str:
        """
        Generate content with a specific Top P setting.
        """
        print(f"🎯 Top P: {top_p} ({purpose})")
        print(f"🌡️ Temperature: {temperature}")
        print("=" * 50)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 40,
                    "max_output_tokens": 400,
                }
            )
            result = response.text.strip()
            self.log_tokens(prompt, result)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def historical_explanation_test(self, top_p: float) -> str:
        """
        Test Top P with historical explanations.
        """
        prompt = """
Explain the significance of the Battle of Plassey (1757) in Indian history. 
Focus on its immediate consequences and long-term impact on British colonial rule.
"""
        return self.generate_with_top_p(
            prompt, 
            top_p, 
            "Historical Explanation Test"
        )
    
    def creative_historical_storytelling(self, top_p: float) -> str:
        """
        Test Top P with creative historical storytelling.
        """
        prompt = """
Describe what it might have been like to be a merchant on the Silk Road during the Tang Dynasty. 
Use vivid descriptions and paint a picture of daily life, challenges, and adventures.
"""
        return self.generate_with_top_p(
            prompt, 
            top_p, 
            "Creative Historical Storytelling",
            temperature=0.8
        )
    
    def analytical_comparison_test(self, top_p: float) -> str:
        """
        Test Top P with analytical historical comparisons.
        """
        prompt = """
Compare the administrative systems of the Mauryan Empire under Chandragupta Maurya 
and the Mughal Empire under Akbar. What were the key similarities and differences?
"""
        return self.generate_with_top_p(
            prompt, 
            top_p, 
            "Analytical Comparison Test",
            temperature=0.5
        )


def demonstrate_top_p_effects():
    """
    Demonstrate how different Top P settings affect various types of historical queries.
    """
    print("🎯 SangamGPT Top P Effects Demonstration")
    print("=" * 70)
    print("Testing how Top P affects diversity and coherence in AI responses")
    print()
    
    explorer = TopPExplorer()
    
    # Test different Top P values
    top_p_values = [
        (0.1, "Very Focused - Only highest probability words"),
        (0.5, "Moderately Focused - Good balance"),
        (0.9, "Balanced - Standard setting for most tasks"),
        (0.99, "Very Diverse - Almost all vocabulary available")
    ]
    
    # Test 1: Historical Explanations
    print("\n📚 TEST 1: HISTORICAL EXPLANATIONS")
    print("=" * 60)
    print("Testing how Top P affects explanation quality and diversity")
    print()
    
    for top_p, description in top_p_values:
        print(f"\n🔸 {description}")
        result = explorer.historical_explanation_test(top_p)
        print(f"Result:\n{result}")
        print("=" * 60)
    
    # Test 2: Creative Storytelling
    print("\n🎨 TEST 2: CREATIVE HISTORICAL STORYTELLING")
    print("=" * 60)
    print("Testing how Top P affects creativity and word diversity")
    print()
    
    for top_p, description in top_p_values[1:]:  # Skip very low Top P for creativity
        print(f"\n🔸 {description}")
        result = explorer.creative_historical_storytelling(top_p)
        print(f"Result:\n{result}")
        print("=" * 60)
    
    # Test 3: Analytical Comparisons
    print("\n🔍 TEST 3: ANALYTICAL HISTORICAL COMPARISONS")
    print("=" * 60)
    print("Testing how Top P affects analytical coherence")
    print()
    
    for top_p, description in top_p_values[:3]:  # Lower Top P for analysis
        print(f"\n🔸 {description}")
        result = explorer.analytical_comparison_test(top_p)
        print(f"Result:\n{result}")
        print("=" * 60)


class TopPComparison:
    """
    Compare the same prompt with different Top P values side by side.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def compare_top_p_side_by_side(self, prompt: str, top_p_values: List[float]) -> Dict[float, str]:
        """
        Run the same prompt with different Top P values for direct comparison.
        """
        results = {}
        
        print(f"🔄 Comparing Same Prompt with Different Top P Values")
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print()
        
        for top_p in top_p_values:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": top_p,
                        "top_k": 40,
                        "max_output_tokens": 300,
                    }
                )
                results[top_p] = response.text.strip()
                
                print(f"🎯 Top P {top_p}:")
                print(f"{results[top_p]}")
                print("-" * 60)
                
            except Exception as e:
                results[top_p] = f"Error: {str(e)}"
                print(f"🎯 Top P {top_p}: Error - {str(e)}")
                print("-" * 60)
        
        return results


def run_side_by_side_comparison():
    """
    Demonstrate the same prompt with different Top P values side by side.
    """
    print("\n📊 SIDE-BY-SIDE TOP P COMPARISON")
    print("=" * 70)
    
    comparator = TopPComparison()
    
    # Historical analysis prompt
    prompt = "Describe the cultural achievements of the Gupta Empire and why it's called the 'Golden Age' of India."
    
    top_p_values_to_test = [0.3, 0.7, 0.95]
    
    results = comparator.compare_top_p_side_by_side(prompt, top_p_values_to_test)
    
    print("\n📋 ANALYSIS OF RESULTS:")
    print("=" * 50)
    print("Notice how:")
    print("• Lower Top P (0.3): More predictable, focused vocabulary")
    print("• Medium Top P (0.7): Good balance of diversity and coherence") 
    print("• Higher Top P (0.95): More diverse word choices and creative expressions")
    print()


def demonstrate_top_p_recommendations():
    """
    Show recommendations for Top P settings in different scenarios.
    """
    print("\n🎯 TOP P RECOMMENDATIONS FOR SANGAMGPT")
    print("=" * 70)
    print("Based on testing, here are optimal Top P settings:")
    print()
    print("📖 Historical Facts & Analysis: Top P 0.3-0.7")
    print("   - Focused vocabulary for accuracy and coherence")
    print()
    print("📚 Educational Explanations: Top P 0.7-0.9")
    print("   - Good balance of clarity and engaging language")
    print()
    print("🎨 Creative Historical Narratives: Top P 0.8-0.95")
    print("   - Diverse vocabulary for vivid storytelling")
    print()
    print("🔍 Analytical Comparisons: Top P 0.5-0.8")
    print("   - Balanced approach for structured analysis")
    print()
    print("💭 Brainstorming & Ideation: Top P 0.9-0.99")
    print("   - Maximum diversity for creative thinking")
    print()


if __name__ == "__main__":
    # Run comprehensive Top P demonstration
    demonstrate_top_p_effects()
    
    # Run side-by-side comparison
    run_side_by_side_comparison()
    
    # Show recommendations
    demonstrate_top_p_recommendations()
    
    print("\n✅ Top P demonstration complete!")
    print("This shows how Top P affects vocabulary diversity and response coherence.")
    print("Top P works together with Temperature to control AI creativity and focus.")
