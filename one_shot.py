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


def get_historical_period(ruler_name):
    """
    Uses one-shot prompting to identify the historical period of a ruler.
    Uses low temperature (0.2) for factual accuracy.
    """
    
    prompt = (
        "You are a historian. Identify the historical period/era of the given ruler.\n\n"
        "Ruler: Julius Caesar\n"
        "Period: Ancient Rome (1st century BCE)\n\n"
        f"Ruler: {ruler_name}\n"
        "Period:"
    )

    print(f"--- Sending One-Shot Prompt to Gemini (Temperature: 0.2) ---\n{prompt}\n---------------------------\n")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,  # Low temperature for factual accuracy
                "max_output_tokens": 150,
                "top_p": 0.8,
                "top_k": 40,
            }
        )
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
