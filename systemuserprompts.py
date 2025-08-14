import os
import google.generativeai as genai
from dotenv import load_dotenv

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


class RTFCFramework:
    """
    RTFC Framework for SangamGPT - Clean implementation with 2 examples
    - Role: Define who the AI should act as
    - Task: Specify what needs to be done  
    - Format: Define response structure
    - Context: Provide background information
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens = 0
    
    def log_tokens(self, prompt: str, response: str):
        """Log token usage."""
        tokens = len((prompt + response).split()) * 0.75
        self.total_tokens += tokens
        print(f"📊 Tokens: {int(tokens)} | Total: {int(self.total_tokens)}")
    
    def create_system_prompt(self, role: str, expertise: list, guidelines: list) -> str:
        """Create system prompt - Role component of RTFC."""
        expertise_str = ", ".join(expertise)
        guidelines_str = "\n".join([f"- {g}" for g in guidelines])
        
        return f"""You are {role} with expertise in {expertise_str}.

EXPERTISE:
{guidelines_str}

STYLE: Professional, accurate, engaging historical content with proper citations."""
    
    def create_user_prompt(self, task: str, context: str, format_req: str, query: str) -> str:
        """Create user prompt - Task, Format, Context components of RTFC."""
        return f"""TASK: {task}
CONTEXT: {context}
FORMAT: {format_req}
QUERY: {query}"""
    
    def generate_with_rtfc(self, role: str, expertise: list, guidelines: list,
                          task: str, context: str, format_req: str, query: str, 
                          temperature: float = 0.6) -> tuple:
        """Generate using complete RTFC framework."""
        
        system_prompt = self.create_system_prompt(role, expertise, guidelines)
        user_prompt = self.create_user_prompt(task, context, format_req, query)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        print(f"🎭 RTFC: {role} | 🌡️ Temp: {temperature}")
        print("=" * 50)
        
        try:
            response = self.model.generate_content(
                combined_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 800,
                }
            )
            ai_response = response.text.strip()
            self.log_tokens(combined_prompt, ai_response)
            return system_prompt, user_prompt, ai_response
            
        except Exception as e:
            return system_prompt, user_prompt, f"Error: {str(e)}"


def example_1_dynasty_analysis():
    """Example 1: Analyzing a historical dynasty using RTFC."""
    print("\n📚 EXAMPLE 1: DYNASTY ANALYSIS")
    print("=" * 60)
    
    rtfc = RTFCFramework()
    
    # RTFC Components
    role = "a specialized historian of Indian dynasties"
    expertise = ["Ancient Indian history", "Political systems", "Cultural developments"]
    guidelines = [
        "Provide chronological analysis with specific dates",
        "Focus on political, cultural, and economic achievements",
        "Include key rulers and their contributions"
    ]
    
    task = "Analyze the rise, peak, and decline of a historical dynasty"
    context = "The user is studying ancient Indian history for educational purposes"
    format_req = "Structured analysis with: Origins, Key Rulers, Major Achievements, Decline Factors"
    query = "Analyze the Chola Dynasty (9th-13th centuries CE)"
    
    system_prompt, user_prompt, response = rtfc.generate_with_rtfc(
        role, expertise, guidelines, task, context, format_req, query, temperature=0.4
    )
    
    print("🎯 AI Response:")
    print(response)
    print("\n" + "="*60)


def example_2_battle_comparison():
    """Example 2: Comparing historical battles using RTFC."""
    print("\n⚔️ EXAMPLE 2: BATTLE COMPARISON")
    print("=" * 60)
    
    rtfc = RTFCFramework()
    
    # RTFC Components  
    role = "a military historian specializing in ancient warfare"
    expertise = ["Military tactics", "Ancient battles", "Strategic analysis"]
    guidelines = [
        "Compare strategies, leadership, and outcomes",
        "Analyze tactical innovations and their effectiveness",
        "Provide historical context for each battle"
    ]
    
    task = "Compare and contrast two significant historical battles"
    context = "Analysis for understanding evolution of military tactics in ancient India"
    format_req = "Comparative analysis with: Background, Tactics, Leadership, Outcomes, Historical Impact"
    query = "Compare the Battle of Kalinga (261 BCE) with the Battle of Tarain (1192 CE)"
    
    system_prompt, user_prompt, response = rtfc.generate_with_rtfc(
        role, expertise, guidelines, task, context, format_req, query, temperature=0.5
    )
    
    print("🎯 AI Response:")
    print(response)
    print("\n" + "="*60)


def demonstrate_rtfc_framework():
    """Demonstrate RTFC framework with clean examples."""
    print("🧠 SangamGPT RTFC Framework Demonstration")
    print("=" * 70)
    print("Showing how RTFC (Role-Task-Format-Context) improves prompting")
    
    example_1_dynasty_analysis()
    example_2_battle_comparison()
    
    print("\n✅ RTFC Framework Benefits:")
    print("- Clear role definition ensures appropriate expertise")
    print("- Specific tasks guide focused responses")  
    print("- Format requirements ensure consistent structure")
    print("- Context provides relevant background for better answers")


if __name__ == "__main__":
    demonstrate_rtfc_framework()
