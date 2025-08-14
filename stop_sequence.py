import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List

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


class StopSequenceExplorer:
    """
    Demonstrates how stop sequences control AI response termination
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
    
    def generate_with_stop_sequence(self, prompt: str, stop_sequences: List[str], purpose: str) -> str:
        """
        Generate content with specific stop sequences.
        """
        print(f"🛑 Stop Sequences: {stop_sequences}")
        print(f"🎯 Purpose: {purpose}")
        print("=" * 50)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                    "max_output_tokens": 500,
                    "stop_sequences": stop_sequences
                }
            )
            result = response.text.strip()
            self.log_tokens(prompt, result)
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    def structured_historical_timeline(self, stop_sequences: List[str]) -> str:
        """
        Example 1: Creating structured historical timeline with stop sequences.
        """
        prompt = """
Create a timeline of the Mughal Empire:

YEAR: 1526
EVENT: Battle of Panipat - Babur defeats Ibrahim Lodi
SIGNIFICANCE: Foundation of Mughal Empire in India

YEAR: 1556
EVENT: Second Battle of Panipat - Akbar secures throne
SIGNIFICANCE: Consolidation of Mughal power

YEAR: 1605
EVENT: Akbar's death, Jahangir becomes emperor
SIGNIFICANCE: Transition to second generation rule

YEAR: 1628
EVENT: Shah Jahan becomes emperor
SIGNIFICANCE: Beginning of architectural golden age

YEAR:"""
        return self.generate_with_stop_sequence(
            prompt,
            stop_sequences,
            "Structured Historical Timeline"
        )
    
    def controlled_historical_analysis(self, stop_sequences: List[str]) -> str:
        """
        Example 2: Controlled analysis with precise stopping points.
        """
        prompt = """
Analyze the factors that led to the decline of the Gupta Empire:

POLITICAL FACTORS:
- Weak succession system led to internal conflicts
- Decentralization of power weakened central authority
- Regional governors gained excessive independence

ECONOMIC FACTORS:
- Decline in trade due to Hunnic invasions
- Reduced tax revenue from disrupted commerce
- Debasement of currency affecting economic stability

MILITARY FACTORS:"""
        return self.generate_with_stop_sequence(
            prompt,
            stop_sequences,
            "Controlled Historical Analysis"
        )


def demonstrate_stop_sequences():
    """
    Demonstrate stop sequence functionality with two clear examples.
    """
    print("🛑 SangamGPT Stop Sequences Demonstration")
    print("=" * 60)
    print("Testing how stop sequences control AI response termination")
    print()
    
    explorer = StopSequenceExplorer()
    
    # Example 1: Timeline with year-based stopping
    print("\n📅 EXAMPLE 1: STRUCTURED TIMELINE")
    print("=" * 50)
    print("Using stop sequence to control timeline generation")
    print()
    
    # Without stop sequence
    print("🔸 WITHOUT Stop Sequence:")
    result1 = explorer.structured_historical_timeline([])
    print(f"Result:\n{result1}")
    print("\n" + "="*50)
    
    # With stop sequence
    print("\n🔸 WITH Stop Sequence ['YEAR:']:")
    result2 = explorer.structured_historical_timeline(["YEAR:"])
    print(f"Result:\n{result2}")
    print("\n" + "="*50)
    
    # Example 2: Analysis with section-based stopping
    print("\n📊 EXAMPLE 2: CONTROLLED ANALYSIS")
    print("=" * 50)
    print("Using stop sequences to control analysis sections")
    print()
    
    # Without stop sequence
    print("🔸 WITHOUT Stop Sequence:")
    result3 = explorer.controlled_historical_analysis([])
    print(f"Result:\n{result3}")
    print("\n" + "="*50)
    
    # With stop sequence
    print("\n🔸 WITH Stop Sequence ['SOCIAL FACTORS:', 'CONCLUSION:']:")
    result4 = explorer.controlled_historical_analysis(["SOCIAL FACTORS:", "CONCLUSION:"])
    print(f"Result:\n{result4}")
    print("\n" + "="*50)


def demonstrate_stop_sequence_benefits():
    """
    Show the benefits of using stop sequences.
    """
    print("\n✅ STOP SEQUENCE BENEFITS:")
    print("=" * 50)
    print("🎯 Precise Control: Stop exactly where you want")
    print("📊 Structured Output: Create consistent formatting")
    print("💰 Cost Efficiency: Save tokens by stopping early")
    print("🔄 Predictable Length: Control response size")
    print("📋 Template Filling: Perfect for forms and structured data")
    print()
    
    print("🎯 BEST USE CASES:")
    print("- Creating structured timelines")
    print("- Generating templated reports")
    print("- Controlling list lengths")
    print("- Section-based content generation")
    print("- Form filling applications")


if __name__ == "__main__":
    # Run the stop sequence demonstration
    demonstrate_stop_sequences()
    
    # Show benefits
    demonstrate_stop_sequence_benefits()
    
    print("\n✅ Stop sequence demonstration complete!")
    print("Stop sequences give you precise control over AI response termination.")
