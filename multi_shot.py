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


def get_dynasty_from_ruler(ruler_name):
    """
    This function takes a ruler's name and uses a multi-shot prompt with the
    Gemini model to identify their dynasty.
    """
    
    # This is our multi-shot prompt
    prompt = (
        "You are an expert historian. Your task is to identify the dynasty of a given historical monarch.\n\n"
        "King: Ashoka the Great\n"
        "Dynasty: Mauryan\n\n"
        "King: Charlemagne\n"
        "Dynasty: Carolingian\n\n"
        "King: Henry VIII\n"
        "Dynasty: Tudor\n\n"
        f"King: {ruler_name}\n"
        "Dynasty:"
    )

    print(f"--- Sending Multi-Shot Prompt to Gemini (Temperature: 0.3) ---\n{prompt}\n---------------------------\n")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,  # Lower temperature for pattern recognition
                "max_output_tokens": 100,
                "top_p": 0.8,
                "top_k": 30,
            }
        )
      
        result = response.text.strip()
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Example Usage ---
ruler_to_identify = "Maravarman Kulasekhara I "

dynasty = get_dynasty_from_ruler(ruler_to_identify)

if dynasty:
    print(f"Gemini Identified Dynasty for {ruler_to_identify}:\n{dynasty}")
