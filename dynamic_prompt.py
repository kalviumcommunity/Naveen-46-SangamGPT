import os
import google.generativeai as genai
from dotenv import load_dotenv
import datetime

load_dotenv()

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
    genai.configure(api_key=api_key)
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    exit()

# Sample historical knowledge base
HISTORICAL_DATA = {
    "Napoleon": {
        "era": "Early 19th century",
        "region": "Europe",
        "type": "Military Leader",
        "key_events": ["Battle of Waterloo", "Napoleonic Wars", "Continental System"]
    },
    "Cleopatra": {
        "era": "1st century BCE",
        "region": "Egypt/Africa",
        "type": "Pharaoh",
        "key_events": ["Alliance with Julius Caesar", "Battle of Actium", "Death by asp"]
    },
    "Ashoka": {
        "era": "3rd century BCE",
        "region": "India",
        "type": "Emperor",
        "key_events": ["Kalinga War", "Buddhist conversion", "Edicts of Ashoka"]
    }
}

def create_dynamic_prompt(historical_figure, user_context="general", depth_level="moderate"):
    """
    Creates a dynamic prompt that adapts based on:
    1. Available knowledge about the figure
    2. User's context/purpose
    3. Desired depth level
    """
    
    # Get current context
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Check if we have specific data about this figure
    figure_data = HISTORICAL_DATA.get(historical_figure, {})
    
    # Build dynamic prompt based on available information
    base_prompt = f"You are an expert historian writing on {current_time}. "
    
    # Adapt based on user context
    if user_context == "student":
        base_prompt += "Explain in simple terms suitable for a student learning history. "
    elif user_context == "academic":
        base_prompt += "Provide a scholarly analysis with historical significance. "
    elif user_context == "tourist":
        base_prompt += "Focus on interesting stories and places to visit related to this figure. "
    else:
        base_prompt += "Provide a balanced historical overview. "
    
    # Adapt based on depth level
    if depth_level == "brief":
        base_prompt += "Keep your response to 2-3 sentences. "
    elif depth_level == "detailed":
        base_prompt += "Provide a comprehensive analysis with multiple paragraphs. "
    else:
        base_prompt += "Provide a moderate-length response. "
    
    # Add specific context if we have data about the figure
    if figure_data:
        context_info = f"""
Context about {historical_figure}:
- Era: {figure_data.get('era', 'Unknown')}
- Region: {figure_data.get('region', 'Unknown')}
- Type: {figure_data.get('type', 'Historical Figure')}
- Key Events: {', '.join(figure_data.get('key_events', ['Various historical events']))}

"""
        base_prompt += context_info
    
    # Add the actual question
    base_prompt += f"\nQuestion: Tell me about {historical_figure}."
    
    return base_prompt

def get_dynamic_historical_info(figure_name, context="general", depth="moderate"):
    """
    Uses dynamic prompting to get information about historical figures.
    """
    
    # Create the dynamic prompt based on inputs
    prompt = create_dynamic_prompt(figure_name, context, depth)
    
    print(f"--- Dynamic Prompt Generated (Temperature: 0.6) ---\n{prompt}\n---------------------------\n")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.6,  # Balanced temperature for dynamic content
                "max_output_tokens": 300,
                "top_p": 0.9,
                "top_k": 40,
            }
        )
        result = response.text.strip()
        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage with different contexts
print("=== EXAMPLE 1: Student Context ===")
result1 = get_dynamic_historical_info("Napoleon", context="student", depth="brief")
if result1:
    print(f"Student-friendly response:\n{result1}\n")

print("=== EXAMPLE 2: Academic Context ===")
result2 = get_dynamic_historical_info("Cleopatra", context="academic", depth="detailed")
if result2:
    print(f"Academic response:\n{result2}\n")

print("=== EXAMPLE 3: Tourist Context ===")
result3 = get_dynamic_historical_info("Ashoka", context="tourist", depth="moderate")
if result3:
    print(f"Tourist-friendly response:\n{result3}\n")
