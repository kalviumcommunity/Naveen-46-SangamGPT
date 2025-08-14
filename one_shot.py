import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
    genai.configure(api_key=api_key)
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    exit()


def log_token_usage(response, prompt_text: str, query_type: str = "One-Shot Query"):
    """
    Log token usage information for the API call.
    """
    try:
        usage_metadata = response.usage_metadata
        
        print(f"\n🪙 TOKEN USAGE - {query_type}")
        print("-" * 40)
        print(f"📝 Prompt Characters: {len(prompt_text)}")
        print(f"📄 Response Characters: {len(response.text)}")
        
        if hasattr(usage_metadata, 'prompt_token_count'):
            print(f"🔤 Input Tokens: {usage_metadata.prompt_token_count}")
        if hasattr(usage_metadata, 'candidates_token_count'):
            print(f"🔤 Output Tokens: {usage_metadata.candidates_token_count}")
        if hasattr(usage_metadata, 'total_token_count'):
            print(f"🔤 Total Tokens: {usage_metadata.total_token_count}")
            # Rough cost estimate
            estimated_cost = usage_metadata.total_token_count * 0.00015 / 1000
            print(f"💰 Estimated Cost: ~${estimated_cost:.6f}")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"⚠️ Token usage not available: {e}")
        print(f"📏 Prompt: {len(prompt_text)} chars, Response: {len(response.text)} chars")
        print("-" * 40)


def get_historical_period(ruler_name):
    """
    Uses one-shot prompting to identify the historical period of a ruler.
    """
    
    prompt = (
        "You are a historian. Identify the historical period/era of the given ruler.\n\n"
        "Ruler: Julius Caesar\n"
        "Period: Ancient Rome (1st century BCE)\n\n"
        f"Ruler: {ruler_name}\n"
        "Period:"
    )

    print(f"--- Sending One-Shot Prompt to Gemini ---\n{prompt}\n---------------------------\n")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        result = response.text.strip()
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
ruler = "akbar"
period = get_historical_period(ruler)

if period:
    print(f"Historical Period for {ruler}:\n{period}")
