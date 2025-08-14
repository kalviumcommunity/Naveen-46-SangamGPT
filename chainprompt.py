import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
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


class ChainOfThoughtAgent:
    """
    An AI agent that uses Chain of Thought prompting to break down complex
    historical questions into step-by-step reasoning processes.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_technological_revolution(self, invention: str, time_period: str) -> str:
        """
        Use Chain of Thought to analyze how technological innovations changed history.
        
        This demonstrates how CoT helps understand technological impact on society.
        """
        
        cot_prompt = f"""
You are a historian specializing in technological history. I want you to think through this step-by-step using Chain of Thought reasoning.

Question: How did the invention of {invention} during {time_period} revolutionize society?

Please work through this systematically:

Step 1: First, let me understand the pre-invention context.
Think: What was life like before {invention}? What problems or limitations existed?

Step 2: Next, I'll examine the invention itself and its initial implementation.
Think: How did {invention} work? Who developed it and under what circumstances?

Step 3: Then, I'll analyze the immediate adoption and early impacts.
Think: How quickly was {invention} adopted? What were the first changes people noticed?

Step 4: I'll examine the broader societal transformations.
Think: How did {invention} change work, social relationships, economic systems, or daily life?

Step 5: I'll consider the unexpected consequences and secondary effects.
Think: What unintended results occurred? How did {invention} enable other changes?

Step 6: Finally, I'll evaluate the long-term revolutionary impact.
Think: How did {invention} fundamentally alter the course of human history?

Please work through each step explicitly, showing your reasoning process.
"""
        
        print(f"⚙️ Analyzing technological revolution using Chain of Thought...")
        print(f"� Invention: {invention}")
        print(f"📅 Period: {time_period}")
        print("=" * 60)
        
        try:
            response = self.model.generate_content(cot_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error in analysis: {str(e)}"
    
    def evaluate_leadership_decisions(self, leader: str, difficult_decision: str) -> str:
        """
        Use Chain of Thought to evaluate controversial historical leadership decisions.
        
        This shows how CoT helps analyze complex moral and strategic choices.
        """
        
        cot_prompt = f"""
You are a historian evaluating leadership decisions. Use Chain of Thought reasoning to analyze this systematically.

Question: How should we evaluate {leader}'s decision regarding {difficult_decision}?

Let me think through this step-by-step:

Step 1: First, I need to understand the historical context and pressures.
Think: What situation was {leader} facing? What constraints, threats, or opportunities existed?

Step 2: I'll examine the available options at the time.
Think: What alternatives did {leader} have? What were the potential consequences of different choices?

Step 3: I'll analyze the decision-making process and reasoning.
Think: What factors did {leader} likely consider? What values or priorities influenced the choice?

Step 4: I'll evaluate the immediate outcomes and results.
Think: What happened immediately after this decision? Did it achieve its intended goals?

Step 5: I'll consider the long-term consequences and legacy.
Think: How did this decision affect future events? What were the lasting impacts?

Step 6: I'll examine different perspectives on this decision.
Think: How did supporters and critics view this choice? How do modern historians assess it?

Step 7: I'll weigh the moral and strategic dimensions.
Think: Was this decision ethically justifiable? Was it strategically sound given the circumstances?

Step 8: I'll reach a balanced historical judgment.
Think: Considering all factors, how should we evaluate this leadership decision?

Please work through each step, showing your reasoning clearly.
"""
        
        print(f"� Evaluating leadership decision using Chain of Thought...")
        print(f"� Leader: {leader}")
        print(f"⚖️ Decision: {difficult_decision}")
        print("=" * 60)
        
        try:
            response = self.model.generate_content(cot_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error in evaluation: {str(e)}"
    
    def trace_cultural_transformation(self, cultural_change: str, society: str) -> str:
        """
        Use Chain of Thought to understand how cultural changes spread through society.
        
        This demonstrates how CoT helps understand cultural evolution and transmission.
        """
        
        cot_prompt = f"""
You are a cultural historian analyzing social transformation. Use Chain of Thought reasoning to understand this change.

Cultural Change to analyze: The spread of {cultural_change} in {society}

Let me work through this systematically:

Step 1: I need to establish the origins and initial context.
Think: Where and when did {cultural_change} first appear? What were the original circumstances?

Step 2: I'll identify the early adopters and catalysts.
Think: Who were the first people or groups to embrace {cultural_change}? What motivated them?

Step 3: I'll examine the mechanisms of cultural transmission.
Think: How did {cultural_change} spread from person to person, group to group? What channels were used?

Step 4: I'll analyze the resistance and acceptance patterns.
Think: Which groups resisted {cultural_change} and why? Which embraced it and what were their reasons?

Step 5: I'll trace the adaptation and modification process.
Think: How did {cultural_change} evolve as it spread? What local variations or adaptations emerged?

Step 6: I'll examine the broader social impacts.
Think: How did {cultural_change} alter social relationships, institutions, or daily life in {society}?

Step 7: I'll consider the long-term cultural legacy.
Think: What lasting effects did {cultural_change} have on {society}? How do we see its influence today?

Step 8: I'll evaluate the transformation process.
Think: What does this case teach us about how cultures change and evolve?

Please think through each step clearly, showing your reasoning process.
"""
        
        print(f"🎭 Analyzing cultural transformation using Chain of Thought...")
        print(f"� Change: {cultural_change}")
        print(f"🏛️ Society: {society}")
        print("=" * 60)
        
        try:
            response = self.model.generate_content(cot_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error in cultural analysis: {str(e)}"
    
    def solve_historical_mystery(self, mystery: str, available_evidence: List[str]) -> str:
        """
        Use Chain of Thought to systematically approach historical mysteries or debates.
        
        This shows how CoT helps structure investigative historical thinking.
        """
        
        evidence_text = "\n".join([f"- {evidence}" for evidence in available_evidence])
        
        cot_prompt = f"""
You are a detective-historian investigating a historical mystery. Use Chain of Thought reasoning to work through this systematically.

Historical Mystery: {mystery}

Available Evidence:
{evidence_text}

Let me approach this like a detective, step by step:

Step 1: I need to clearly define what we're trying to determine.
Think: What exactly is the question or mystery we're solving? What would constitute an answer?

Step 2: I'll examine each piece of evidence systematically.
Think: What does each piece of evidence tell us? What are its strengths and limitations?

Step 3: I'll look for patterns and connections in the evidence.
Think: How do different pieces of evidence relate to each other? Do they support or contradict each other?

Step 4: I'll consider what evidence might be missing.
Think: What additional information would help solve this mystery? Why might this evidence be unavailable?

Step 5: I'll evaluate different possible explanations or theories.
Think: What are the main competing theories or explanations? How well does each fit the evidence?

Step 6: I'll assess the reliability and bias of sources.
Think: Who created this evidence and why? What biases or limitations might affect it?

Step 7: I'll weigh the evidence and draw conclusions.
Think: Based on all the evidence, what is the most likely explanation? How confident can we be?

Step 8: I'll acknowledge uncertainties and limitations.
Think: What aspects remain unclear? What are the limits of what we can know?

Please work through this investigation step by step, showing your reasoning clearly.
"""
        
        print(f"🔍 Investigating historical mystery using Chain of Thought...")
        print(f"❓ Mystery: {mystery}")
        print(f"📋 Evidence pieces: {len(available_evidence)}")
        print("=" * 60)
        
        try:
            response = self.model.generate_content(cot_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error in mystery investigation: {str(e)}"


def demonstrate_chain_of_thought():
    """
    Demonstrate various Chain of Thought prompting techniques with historical examples.
    """
    print("🧠 SangamGPT Chain of Thought Prompting Demo")
    print("=" * 60)
    
    agent = ChainOfThoughtAgent()
    
    # Example 1: Technological Revolution Analysis
    print("\n🎯 EXAMPLE 1: Technological Revolution Analysis")
    print("=" * 40)
    
    result1 = agent.analyze_technological_revolution(
        "the printing press", 
        "Renaissance Europe"
    )
    
    print("✨ Chain of Thought Analysis:")
    print(result1)
    
    # Example 2: Leadership Decision Evaluation
    print("\n\n🎯 EXAMPLE 2: Leadership Decision Evaluation")
    print("=" * 40)
    
    result2 = agent.evaluate_leadership_decisions(
        "Thomas Jefferson",
        "the Louisiana Purchase in 1803"
    )
    
    print("✨ Chain of Thought Evaluation:")
    print(result2)
    
    # Example 3: Cultural Transformation Analysis
    print("\n\n🎯 EXAMPLE 3: Cultural Transformation Analysis")
    print("=" * 40)
    
    result3 = agent.trace_cultural_transformation(
        "Buddhism", 
        "ancient Asia"
    )
    
    print("✨ Chain of Thought Cultural Analysis:")
    print(result3)
    
    # Example 4: Historical Scenario Reconstruction
    print("\n\n🎯 EXAMPLE 4: Historical Scenario Reconstruction")
    print("=" * 40)
    
    print("✨ Chain of Thought Reconstruction:")
    print(result4)


if __name__ == "__main__":
    demonstrate_chain_of_thought()
