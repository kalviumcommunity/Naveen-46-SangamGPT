import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini API with the key from the environment
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
    genai.configure(api_key=api_key)
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    print("Please make sure you have a .env file with your GOOGLE_API_KEY.")
    exit()


def get_king_summary(king_name):
    """
    This function takes a king's name and uses a zero-shot prompt with the
    Gemini model to generate a brief summary of their reign.
    Uses medium temperature (0.5) for informative but engaging content.
    """
    
    # This is our zero-shot prompt for Gemini. We are asking the model to generate
    prompt = (
        "You are an expert historian. Provide a brief, one-paragraph summary of the "
        f"reign and major achievements of the following monarch: {king_name}"
    )

    print(f"--- Sending Zero-Shot Prompt to Gemini (Temperature: 0.5) ---\n{prompt}\n---------------------------\n")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Send the prompt to the model with temperature settings
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.5,  # Medium temperature for informative content
                "max_output_tokens": 200,
                "top_p": 0.8,
                "top_k": 40,
            }
        )
        
        # Extract the text from the response
        result = response.text.strip()
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

king_to_summarize = "Rajendra Cholan"

# We call our function with the king's name.
summary = get_king_summary(king_to_summarize)

if summary:
    print(f"Gemini Summary for {king_to_summarize}:\n{summary}")

