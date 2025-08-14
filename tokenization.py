import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import time

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


class TokenizationAnalyzer:
    """
    A comprehensive tool to analyze and demonstrate token usage patterns
    in different types of historical AI queries.
    """
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.query_count = 0
    
    def log_detailed_token_usage(self, response, prompt_text: str, query_type: str) -> Dict[str, Any]:
        """
        Log comprehensive token usage information and return metrics.
        """
        try:
            usage_metadata = response.usage_metadata
            
            # Calculate metrics
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = getattr(usage_metadata, 'total_token_count', input_tokens + output_tokens)
            
            # Rough cost estimate (Gemini pricing approximation)
            cost_estimate = total_tokens * 0.00015 / 1000
            
            # Update running totals
            self.total_tokens_used += total_tokens
            self.total_cost_estimate += cost_estimate
            self.query_count += 1
            
            print(f"\n🪙 TOKEN ANALYSIS - {query_type}")
            print("=" * 50)
            print(f"📝 Prompt Length: {len(prompt_text)} characters")
            print(f"📄 Response Length: {len(response.text)} characters")
            print(f"🔤 Input Tokens: {input_tokens}")
            print(f"🔤 Output Tokens: {output_tokens}")
            print(f"🔤 Total Tokens: {total_tokens}")
            print(f"💰 Query Cost: ~${cost_estimate:.6f}")
            print(f"📊 Efficiency: {len(response.text)/total_tokens:.2f} chars/token")
            print("=" * 50)
            
            return {
                'query_type': query_type,
                'prompt_chars': len(prompt_text),
                'response_chars': len(response.text),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost_estimate': cost_estimate,
                'efficiency': len(response.text)/total_tokens if total_tokens > 0 else 0
            }
            
        except Exception as e:
            print(f"⚠️ Token logging error: {e}")
            return {'error': str(e)}
    
    def analyze_query_types(self):
        """
        Test different types of historical queries and analyze their token usage.
        """
        print("🧪 TOKENIZATION ANALYSIS: Different Query Types")
        print("=" * 60)
        
        queries = [
            {
                'prompt': "When did Akbar rule?",
                'type': "Simple Fact Query",
                'expected_tokens': "Low (5-20 total)"
            },
            {
                'prompt': "Provide a detailed analysis of Ashoka's transformation after the Kalinga War, including its impact on his policies and the spread of Buddhism.",
                'type': "Complex Analysis Query", 
                'expected_tokens': "High (200-500 total)"
            },
            {
                'prompt': "List the Mughal emperors in chronological order with their reign dates.",
                'type': "Structured Information Query",
                'expected_tokens': "Medium (50-150 total)"
            },
            {
                'prompt': "Write a creative story about a day in the life of a craftsman during the Chola period. Make it engaging and historically accurate.",
                'type': "Creative Narrative Query",
                'expected_tokens': "Very High (300-800 total)"
            },
            {
                'prompt': "Compare Chandragupta Maurya and Harshavadhana: leadership style, military strategy, administrative system.",
                'type': "Comparative Analysis Query",
                'expected_tokens': "High (200-400 total)"
            }
        ]
        
        results = []
        
        for query in queries:
            print(f"\n🎯 Testing: {query['type']}")
            print(f"📝 Prompt: \"{query['prompt']}\"")
            print(f"🔮 Expected: {query['expected_tokens']}")
            
            try:
                start_time = time.time()
                response = self.model.generate_content(
                    query['prompt'],
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 500,
                    }
                )
                end_time = time.time()
                
                # Log token usage
                metrics = self.log_detailed_token_usage(response, query['prompt'], query['type'])
                metrics['response_time'] = end_time - start_time
                results.append(metrics)
                
                # Show partial response
                response_preview = response.text[:100] + "..." if len(response.text) > 100 else response.text
                print(f"📄 Response Preview: {response_preview}")
                print(f"⏱️ Response Time: {metrics['response_time']:.2f} seconds")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({'error': str(e), 'query_type': query['type']})
        
        return results
    
    def analyze_language_differences(self):
        """
        Compare token usage across different languages for the same concept.
        """
        print(f"\n🌐 LANGUAGE TOKENIZATION COMPARISON")
        print("=" * 60)
        
        concept_queries = [
            {
                'english': "Hello, how are you?",
                'hindi': "नमस्ते, आप कैसे हैं?",
                'tamil': "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
                'concept': "Greeting"
            },
            {
                'english': "The Mughal Empire was established in India.",
                'hindi': "मुगल साम्राज्य की स्थापना भारत में हुई थी।",
                'tamil': "முகலாய சாம்ராஜ்யம் இந்தியாவில் நிறுவப்பட்டது.",
                'concept': "Historical Fact"
            }
        ]
        
        for concept in concept_queries:
            print(f"\n🎭 Concept: {concept['concept']}")
            print("-" * 40)
            
            for lang, text in concept.items():
                if lang == 'concept':
                    continue
                    
                try:
                    response = self.model.generate_content(
                        f"Translate this to English and explain: {text}",
                        generation_config={"max_output_tokens": 100}
                    )
                    
                    metrics = self.log_detailed_token_usage(response, text, f"{lang.capitalize()} Query")
                    
                except Exception as e:
                    print(f"❌ Error with {lang}: {e}")
    
    def analyze_prompt_efficiency(self):
        """
        Compare efficient vs inefficient prompts for the same information.
        """
        print(f"\n⚡ PROMPT EFFICIENCY COMPARISON")
        print("=" * 60)
        
        prompt_pairs = [
            {
                'inefficient': "I would really appreciate it if you could please help me understand the historical significance and importance of the Mauryan Empire in ancient Indian history, and I want you to provide me with detailed information about this very important historical topic.",
                'efficient': "Explain the historical significance of the Mauryan Empire.",
                'topic': "Historical Significance"
            },
            {
                'inefficient': "Could you please tell me, in your own words and with great detail, about what happened during the reign of Emperor Ashoka, particularly focusing on the events and consequences of the Kalinga War?",
                'efficient': "Describe Ashoka's reign and the impact of the Kalinga War.",
                'topic': "Ashoka's Reign"
            }
        ]
        
        for pair in prompt_pairs:
            print(f"\n📊 Topic: {pair['topic']}")
            print("-" * 40)
            
            for efficiency, prompt in pair.items():
                if efficiency == 'topic':
                    continue
                    
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "max_output_tokens": 200,
                            "temperature": 0.5
                        }
                    )
                    
                    metrics = self.log_detailed_token_usage(response, prompt, f"{efficiency.capitalize()} Prompt")
                    
                except Exception as e:
                    print(f"❌ Error with {efficiency} prompt: {e}")
    
    def print_session_summary(self):
        """
        Print a summary of the entire tokenization analysis session.
        """
        print(f"\n📋 SESSION SUMMARY")
        print("=" * 60)
        print(f"🔢 Total Queries: {self.query_count}")
        print(f"🪙 Total Tokens Used: {self.total_tokens_used:,}")
        print(f"💰 Total Estimated Cost: ${self.total_cost_estimate:.6f}")
        print(f"📊 Average Tokens per Query: {self.total_tokens_used/self.query_count:.1f}" if self.query_count > 0 else "📊 No queries processed")
        print(f"💵 Average Cost per Query: ${self.total_cost_estimate/self.query_count:.6f}" if self.query_count > 0 else "💵 No cost data")
        
        # Token usage recommendations
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 40)
        if self.total_tokens_used > 1000:
            print("⚠️  High token usage detected. Consider:")
            print("   • Using more concise prompts")
            print("   • Reducing max_output_tokens")
            print("   • Batching similar queries")
        else:
            print("✅ Token usage is reasonable for this session")
        
        print(f"💡 Cost optimization tips:")
        print("   • Track tokens per query type")
        print("   • Use structured outputs when possible")  
        print("   • Monitor daily/monthly usage")


def run_comprehensive_tokenization_analysis():
    """
    Run a complete tokenization analysis for the SangamGPT project.
    """
    print("🪙 SANGAMGPT TOKENIZATION ANALYSIS")
    print("=" * 60)
    print("Analyzing token usage patterns across different query types")
    print("This helps optimize costs and improve efficiency\n")
    
    analyzer = TokenizationAnalyzer()
    
    # Test different query types
    print("Phase 1: Query Type Analysis")
    query_results = analyzer.analyze_query_types()
    
    # Test language differences  
    print("\nPhase 2: Language Comparison")
    analyzer.analyze_language_differences()
    
    # Test prompt efficiency
    print("\nPhase 3: Prompt Efficiency")
    analyzer.analyze_prompt_efficiency()
    
    # Print session summary
    analyzer.print_session_summary()
    
    print(f"\n✅ Tokenization analysis complete!")
    print("This data helps you understand and optimize AI costs in SangamGPT")


if __name__ == "__main__":
    run_comprehensive_tokenization_analysis()
