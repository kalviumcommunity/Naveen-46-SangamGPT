# Structured Output example with Google Gemini
# pip install google-generativeai python-dotenv

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment.")
    genai.configure(api_key=api_key)
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
    exit()

def get_structured_historical_info(historical_figure: str) -> Optional[Dict]:
    """
    Uses structured output to get historical information in a predefined JSON format.
    This ensures the AI response can be easily processed by our application.
    """
    
    # Define the exact JSON structure we want
    json_schema = {
        "name": "string - Full name of the historical figure",
        "birth_year": "number - Year of birth (use negative numbers for BCE)",
        "death_year": "number - Year of death (use negative numbers for BCE)",
        "dynasty_or_empire": "string - The dynasty, empire, or political entity they belonged to",
        "region": "string - Geographic region or modern country",
        "title": "string - Their primary title or role",
        "major_achievements": "array of strings - List of 3-5 key achievements",
        "historical_significance": "string - One sentence about their importance",
        "famous_quote": "string - A notable quote attributed to them, or 'None known' if unavailable"
    }
    
    # Create a prompt that enforces structured output
    prompt = f"""
You are a historian providing structured data about historical figures. 
You MUST respond with valid JSON that follows this exact schema:

{json.dumps(json_schema, indent=2)}

Provide information about: {historical_figure}

Your response must be valid JSON only, no additional text or explanation.
If you don't know a specific piece of information, use "Unknown" for strings or 0 for numbers.
"""

    print(f"--- Requesting Structured Output for {historical_figure} ---")
    print(f"Expected JSON Schema:\n{json.dumps(json_schema, indent=2)}")
    print("---------------------------\n")

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # Extract and parse JSON response
        json_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if json_text.startswith("```json"):
            json_text = json_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON to validate structure
        structured_data = json.loads(json_text)
        
        return structured_data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def display_structured_data(data: Dict) -> None:
    """
    Display the structured data in a formatted way.
    """
    if not data:
        print("No data to display")
        return
    
    print("=== STRUCTURED HISTORICAL DATA ===")
    print(f"Name: {data.get('name', 'Unknown')}")
    print(f"Lifespan: {data.get('birth_year', 'Unknown')} - {data.get('death_year', 'Unknown')}")
    print(f"Title: {data.get('title', 'Unknown')}")
    print(f"Dynasty/Empire: {data.get('dynasty_or_empire', 'Unknown')}")
    print(f"Region: {data.get('region', 'Unknown')}")
    print(f"Historical Significance: {data.get('historical_significance', 'Unknown')}")
    
    print("\nMajor Achievements:")
    achievements = data.get('major_achievements', [])
    for i, achievement in enumerate(achievements, 1):
        print(f"  {i}. {achievement}")
    
    print(f"\nFamous Quote: \"{data.get('famous_quote', 'None known')}\"")
    print("=" * 50)

def save_to_database_simulation(data: Dict) -> None:
    """
    Simulate saving structured data to a database.
    In a real application, this would insert into a database.
    """
    print("\n--- SIMULATING DATABASE SAVE ---")
    print("INSERT INTO historical_figures (")
    print("  name, birth_year, death_year, title, dynasty_or_empire, region")
    print(") VALUES (")
    print(f"  '{data.get('name')}', {data.get('birth_year')}, {data.get('death_year')},")
    print(f"  '{data.get('title')}', '{data.get('dynasty_or_empire')}', '{data.get('region')}'")
    print(");")
    print("Database save completed!")

# Example usage
if __name__ == "__main__":
    # Test with multiple historical figures
    historical_figures = ["Chandragupta Maurya", "Pandiyan"]
    
    for figure in historical_figures:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {figure}")
        print('='*60)
        
        # Get structured data
        structured_info = get_structured_historical_info(figure)
        
        if structured_info:
            # Display the data
            display_structured_data(structured_info)
            
            # Simulate database operations
            save_to_database_simulation(structured_info)
            
            # Show raw JSON for debugging
            print(f"\nRaw JSON:\n{json.dumps(structured_info, indent=2)}")
        else:
            print(f"Failed to get structured data for {figure}")
        
        print("\n" + "="*60)
