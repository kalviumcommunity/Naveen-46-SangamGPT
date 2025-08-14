import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

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


class TemperatureExplorer:
    """
    Demonstrates how different temperature settings affect AI responses
    in historical research contexts.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_with_temperature(self, prompt: str, temperature: float, purpose: str, top_k: int = 60, stop_sequences: List[str] = None) -> str:
        """
        Generate content with temperature, Top K, and stop sequence settings.
        """
        print(f"🌡️ Temperature: {temperature} ({purpose})")
        print(f"🔢 Top K: {top_k}")
        if stop_sequences:
            print(f"🛑 Stop Sequences: {stop_sequences}")
        print("=" * 50)
        
        config = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": 0.9,
            "max_output_tokens": 500,
        }
        
        if stop_sequences:
            config["stop_sequences"] = stop_sequences
        
        try:
            response = self.model.generate_content(prompt, generation_config=config)
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    def factual_historical_query(self, temperature: float) -> str:
        """
        Test temperature with factual historical questions.
        """
        prompt = """
What year did the Roman Empire fall? Provide specific dates and key events.
"""
        return self.generate_with_temperature(
            prompt, 
            temperature, 
            "Factual Historical Query",
            top_k=30,  # Lower Top K for factual accuracy
            stop_sequences=["In conclusion", "Summary:", "Note:"]  # Stop at analysis end
        )
    
    def creative_historical_narrative(self, temperature: float) -> str:
        """
        Test temperature with creative historical storytelling.
        """
        prompt = """
Write a creative description of what it might have felt like to be a Roman citizen on the day the empire fell. Use vivid imagery and emotional language.
"""
        return self.generate_with_temperature(
            prompt, 
            temperature, 
            "Creative Historical Narrative",
            top_k=120,  # Higher Top K for creative vocabulary
            stop_sequences=["THE END", "---", "EPILOGUE:"]  # Natural story endings
        )
    
    def analytical_comparison(self, temperature: float) -> str:
        """
        Test temperature with analytical historical comparison.
        """
        prompt = """
Compare and contrast the military strategies of Alexander the Great and Julius Caesar. Focus on their tactical approaches and leadership styles.
"""
        return self.generate_with_temperature(
            prompt, 
            temperature, 
            "Analytical Comparison",
            top_k=70  # Medium Top K for analytical balance
        )
    
    def historical_brainstorming(self, temperature: float) -> str:
        """
        Test temperature with creative historical brainstorming.
        """
        prompt = """
Generate 5 creative "what if" scenarios about ancient Indian history. Think of alternative historical outcomes and their potential impacts.
"""
        return self.generate_with_temperature(
            prompt, 
            temperature, 
            "Historical Brainstorming",
            top_k=200  # High Top K for maximum creative vocabulary
        )


def demonstrate_temperature_effects():
    """
    Demonstrate how different temperature settings affect various types of historical queries.
    """
    print("🌡️ SangamGPT Temperature Effects Demonstration")
    print("=" * 60)
    print("Testing how temperature affects different types of historical AI responses")
    print()
    
    explorer = TemperatureExplorer()
    
    # Test different temperature ranges
    temperatures = [
        (0.1, "Very Conservative - Maximum Precision"),
        (0.3, "Conservative - Reliable and Focused"),
        (0.7, "Balanced - Creative but Controlled"),
        (0.9, "Creative - High Variability"),
        (1.2, "Very Creative - Maximum Diversity")
    ]
    
    # Test 1: Factual Queries
    print("\n📚 TEST 1: FACTUAL HISTORICAL QUERIES")
    print("=" * 60)
    print("Testing how temperature affects factual accuracy and consistency")
    print()
    
    for temp, description in temperatures[:3]:  # Only test lower temperatures for facts
        result = explorer.factual_historical_query(temp)
        print(f"Result:\n{result}\n")
        print("-" * 40)
    
    # Test 2: Creative Narratives
    print("\n🎨 TEST 2: CREATIVE HISTORICAL NARRATIVES")
    print("=" * 60)
    print("Testing how temperature affects creativity and storytelling")
    print()
    
    for temp, description in temperatures[1:4]:  # Test mid to high temperatures
        result = explorer.creative_historical_narrative(temp)
        print(f"Result:\n{result}\n")
        print("-" * 40)
    
    # Test 3: Analytical Tasks
    print("\n🔍 TEST 3: ANALYTICAL HISTORICAL COMPARISONS")
    print("=" * 60)
    print("Testing how temperature affects analytical depth and consistency")
    print()
    
    for temp, description in temperatures[:3]:  # Lower temperatures for analysis
        result = explorer.analytical_comparison(temp)
        print(f"Result:\n{result}\n")
        print("-" * 40)
    
    # Test 4: Brainstorming
    print("\n💡 TEST 4: HISTORICAL BRAINSTORMING")
    print("=" * 60)
    print("Testing how temperature affects creative ideation and diversity")
    print()
    
    for temp, description in temperatures[2:]:  # Higher temperatures for creativity
        result = explorer.historical_brainstorming(temp)
        print(f"Result:\n{result}\n")
        print("-" * 40)
    
    # Recommendations
    print("\n🎯 TEMPERATURE RECOMMENDATIONS FOR SANGAMGPT")
    print("=" * 60)
    print("Based on testing, here are optimal temperature settings:")
    print()
    print("📖 Historical Facts & Dates: Temperature 0.1-0.2")
    print("   - Maximum accuracy and consistency needed")
    print()
    print("🔍 Historical Analysis: Temperature 0.3-0.5") 
    print("   - Balance between accuracy and explanatory variety")
    print()
    print("📚 Educational Content: Temperature 0.5-0.7")
    print("   - Engaging but reliable educational material")
    print()
    print("🎨 Creative Storytelling: Temperature 0.7-0.9")
    print("   - Vivid narratives and engaging descriptions")
    print()
    print("💭 Brainstorming Ideas: Temperature 0.8-1.0")
    print("   - Maximum creativity for generating diverse ideas")
    print()


class TemperatureComparison:
    """
    A class to directly compare the same prompt with different temperatures.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def compare_temperatures_side_by_side(self, prompt: str, temperatures: List[float]) -> Dict[float, str]:
        """
        Run the same prompt with different temperatures for direct comparison.
        """
        results = {}
        
        print(f"🔄 Comparing Same Prompt with Different Temperatures")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print()
        
        for temp in temperatures:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temp,
                        "top_k": 60,  # Consistent Top K for temperature comparison
                        "top_p": 0.8,
                        "max_output_tokens": 300,
                    }
                )
                results[temp] = response.text.strip()
                
                print(f"🌡️ Temperature {temp}:")
                print(f"{results[temp]}")
                print("-" * 50)
                
            except Exception as e:
                results[temp] = f"Error: {str(e)}"
                print(f"🌡️ Temperature {temp}: Error - {str(e)}")
                print("-" * 50)
        
        return results


def run_side_by_side_comparison():
    """
    Demonstrate the same prompt with different temperatures side by side.
    """
    print("\n📊 SIDE-BY-SIDE TEMPERATURE COMPARISON")
    print("=" * 60)
    
    comparator = TemperatureComparison()
    
    # Historical explanation prompt
    prompt = "Explain why the Mughal Empire declined in India. Keep it concise but informative."
    
    temperatures_to_test = [0.2, 0.5, 0.8]
    
    results = comparator.compare_temperatures_side_by_side(prompt, temperatures_to_test)
    
    print("\n📋 ANALYSIS OF RESULTS:")
    print("=" * 40)
    print("Notice how:")
    print("• Lower temperature (0.2): More focused, consistent, factual")
    print("• Medium temperature (0.5): Good balance of facts and readability") 
    print("• Higher temperature (0.8): More varied language, creative expressions")
    print()


if __name__ == "__main__":
    # Run the comprehensive temperature demonstration
    demonstrate_temperature_effects()
    
    # Run side-by-side comparison
    run_side_by_side_comparison()
    
    print("\n✅ Temperature demonstration complete!")
    print("This shows how temperature affects AI responses in historical research contexts.")
