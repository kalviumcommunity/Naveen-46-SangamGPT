import os
import json
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

# Historical Database - In production, this would be a real database
HISTORICAL_DB = {
    "rulers": {
        "ashoka": {
            "name": "Ashoka the Great",
            "dynasty": "Mauryan Empire", 
            "reign": {"start": -268, "end": -232},
            "capital": "Pataliputra",
            "region": "Indian subcontinent",
            "achievements": ["Spread of Buddhism", "Edicts of Ashoka", "Unification of India"]
        },
        "akbar": {
            "name": "Akbar the Great",
            "dynasty": "Mughal Empire",
            "reign": {"start": 1556, "end": 1605}, 
            "capital": "Fatehpur Sikri",
            "region": "Indian subcontinent",
            "achievements": ["Religious tolerance", "Administrative reforms", "Cultural synthesis"]
        },
        "napoleon": {
            "name": "Napoleon Bonaparte", 
            "dynasty": "First French Empire",
            "reign": {"start": 1804, "end": 1814},
            "capital": "Paris",
            "region": "Europe", 
            "achievements": ["Napoleonic Code", "Continental System", "Military innovations"]
        }
    },
    "battles": {
        "kalinga_war": {
            "name": "Kalinga War",
            "year": -261,
            "location": "Kalinga (modern Odisha)",
            "belligerents": ["Mauryan Empire", "Kalinga Kingdom"],
            "outcome": "Mauryan victory",
            "significance": "Led to Ashoka's conversion to Buddhism"
        },
        "battle_of_waterloo": {
            "name": "Battle of Waterloo", 
            "year": 1815,
            "location": "Waterloo, Belgium",
            "belligerents": ["French Empire", "Seventh Coalition"],
            "outcome": "Coalition victory",
            "significance": "End of Napoleon's Hundred Days"
        }
    }
}

class HistoricalDatabase:
    """
    A class to handle all database operations for historical data.
    In a real application, this would connect to PostgreSQL, MongoDB, etc.
    """
    
    def __init__(self):
        self.data = HISTORICAL_DB
    
    def search_ruler(self, name: str) -> Optional[Dict]:
        """Search for a ruler by name (case-insensitive)"""
        name_key = name.lower().replace(" ", "_")
        
        # Try exact match first
        if name_key in self.data["rulers"]:
            return self.data["rulers"][name_key]
        
        # Try partial matches
        for key, ruler_data in self.data["rulers"].items():
            if name.lower() in ruler_data["name"].lower():
                return ruler_data
        
        return None
    
    def search_battle(self, name: str) -> Optional[Dict]:
        """Search for a battle by name (case-insensitive)"""
        name_key = name.lower().replace(" ", "_")
        
        if name_key in self.data["battles"]:
            return self.data["battles"][name_key]
        
        # Try partial matches
        for key, battle_data in self.data["battles"].items():
            if name.lower() in battle_data["name"].lower():
                return battle_data
        
        return None
    
    def get_rulers_by_period(self, start_year: int, end_year: int) -> List[Dict]:
        """Get all rulers who reigned during the specified period"""
        rulers = []
        
        for ruler_data in self.data["rulers"].values():
            reign_start = ruler_data["reign"]["start"]
            reign_end = ruler_data["reign"]["end"]
            
            # Check if reign overlaps with the requested period
            if reign_start <= end_year and reign_end >= start_year:
                rulers.append(ruler_data)
        
        return rulers

# Initialize our database
db = HistoricalDatabase()


def get_ruler_info(ruler_name: str) -> Dict[str, Any]:
    """
    Retrieve comprehensive information about a historical ruler.
    
    Args:
        ruler_name: Name of the ruler to search for
        
    Returns:
        Dictionary with ruler information or error message
    """
    print(f"� Looking up ruler: {ruler_name}")
    
    ruler_data = db.search_ruler(ruler_name)
    
    if ruler_data:
        # Format the years properly
        start_year = ruler_data["reign"]["start"]
        end_year = ruler_data["reign"]["end"]
        
        if start_year < 0:
            start_str = f"{abs(start_year)} BCE"
        else:
            start_str = f"{start_year} CE"
            
        if end_year < 0:
            end_str = f"{abs(end_year)} BCE"
        else:
            end_str = f"{end_year} CE"
        
        return {
            "success": True,
            "name": ruler_data["name"],
            "dynasty": ruler_data["dynasty"],
            "reign_period": f"{start_str} - {end_str}",
            "capital": ruler_data["capital"],
            "region": ruler_data["region"],
            "major_achievements": ruler_data["achievements"],
            "reign_duration": abs(end_year - start_year)
        }
    else:
        return {
            "success": False,
            "message": f"Sorry, I couldn't find information about '{ruler_name}' in our database.",
            "suggestion": "Try searching for: Ashoka, Akbar, or Napoleon"
        }


def get_battle_info(battle_name: str) -> Dict[str, Any]:
    """
    Retrieve information about a historical battle.
    
    Args:
        battle_name: Name of the battle to search for
        
    Returns:
        Dictionary with battle information or error message
    """
    print(f"⚔️ Looking up battle: {battle_name}")
    
    battle_data = db.search_battle(battle_name)
    
    if battle_data:
        year = battle_data["year"]
        year_str = f"{abs(year)} BCE" if year < 0 else f"{year} CE"
        
        return {
            "success": True,
            "name": battle_data["name"],
            "year": year_str,
            "location": battle_data["location"],
            "belligerents": battle_data["belligerents"],
            "outcome": battle_data["outcome"],
            "historical_significance": battle_data["significance"]
        }
    else:
        return {
            "success": False,
            "message": f"Sorry, I couldn't find information about '{battle_name}' in our database.",
            "suggestion": "Try searching for: Kalinga War or Battle of Waterloo"
        }


def get_timeline_events(start_year: int, end_year: int) -> Dict[str, Any]:
    """
    Get historical events (rulers and battles) within a specific time range.
    
    Args:
        start_year: Start year (negative for BCE)
        end_year: End year (negative for BCE)
        
    Returns:
        Dictionary with timeline events
    """
    print(f"📅 Searching timeline: {start_year} to {end_year}")
    
    if start_year > end_year:
        return {
            "success": False,
            "message": "Start year must be less than or equal to end year"
        }
    
    # Get rulers in this period
    rulers = db.get_rulers_by_period(start_year, end_year)
    
    # Get battles in this period
    battles = []
    for battle_data in db.data["battles"].values():
        battle_year = battle_data["year"]
        if start_year <= battle_year <= end_year:
            battles.append(battle_data)
    
    # Format the response
    events = []
    
    for ruler in rulers:
        start_year_str = f"{abs(ruler['reign']['start'])} BCE" if ruler['reign']['start'] < 0 else f"{ruler['reign']['start']} CE"
        end_year_str = f"{abs(ruler['reign']['end'])} BCE" if ruler['reign']['end'] < 0 else f"{ruler['reign']['end']} CE"
        
        events.append({
            "type": "ruler",
            "name": ruler["name"],
            "period": f"{start_year_str} - {end_year_str}",
            "dynasty": ruler["dynasty"]
        })
    
    for battle in battles:
        year_str = f"{abs(battle['year'])} BCE" if battle['year'] < 0 else f"{battle['year']} CE"
        events.append({
            "type": "battle", 
            "name": battle["name"],
            "year": year_str,
            "location": battle["location"]
        })
    
    return {
        "success": True,
        "time_range": f"{abs(start_year)} {'BCE' if start_year < 0 else 'CE'} to {abs(end_year)} {'BCE' if end_year < 0 else 'CE'}",
        "total_events": len(events),
        "events": sorted(events, key=lambda x: x.get('year', x.get('period', '')))
    }

class FunctionCallHandler:
    """
    Handles the detection and execution of function calls in AI responses.
    This would typically be more sophisticated in production.
    """
    
    def __init__(self):
        self.available_functions = {
            "get_ruler_info": {
                "function": get_ruler_info,
                "description": "Get detailed information about a historical ruler",
                "parameters": {"ruler_name": "Name of the historical ruler"}
            },
            "get_battle_info": {
                "function": get_battle_info,
                "description": "Get information about a historical battle",
                "parameters": {"battle_name": "Name of the historical battle"}
            },
            "get_timeline_events": {
                "function": get_timeline_events,
                "description": "Get historical events within a time period",
                "parameters": {
                    "start_year": "Start year (negative for BCE)",
                    "end_year": "End year (negative for BCE)"
                }
            }
        }
    
    def get_function_descriptions(self) -> str:
        """Generate a description of available functions for the AI prompt"""
        descriptions = []
        for func_name, func_info in self.available_functions.items():
            params = ", ".join([f"{p}: {desc}" for p, desc in func_info["parameters"].items()])
            descriptions.append(f"- {func_name}({params}): {func_info['description']}")
        
        return "\n".join(descriptions)
    
    def detect_function_calls(self, response: str) -> List[Dict]:
        """
        Parse AI response to detect function calls.
        In production, you'd use more robust parsing or JSON-based function calling.
        """
        function_calls = []
        
        # Look for function call patterns
        import re
        
        # Pattern for get_ruler_info("name")
        ruler_pattern = r'get_ruler_info\(["\']([^"\']+)["\']\)'
        ruler_matches = re.findall(ruler_pattern, response)
        for match in ruler_matches:
            function_calls.append({
                "function": "get_ruler_info",
                "parameters": {"ruler_name": match}
            })
        
        # Pattern for get_battle_info("name")
        battle_pattern = r'get_battle_info\(["\']([^"\']+)["\']\)'
        battle_matches = re.findall(battle_pattern, response)
        for match in battle_matches:
            function_calls.append({
                "function": "get_battle_info", 
                "parameters": {"battle_name": match}
            })
        
        # Pattern for get_timeline_events(start, end)
        timeline_pattern = r'get_timeline_events\((-?\d+),\s*(-?\d+)\)'
        timeline_matches = re.findall(timeline_pattern, response)
        for start, end in timeline_matches:
            function_calls.append({
                "function": "get_timeline_events",
                "parameters": {
                    "start_year": int(start),
                    "end_year": int(end)
                }
            })
        
        return function_calls
    
    def execute_functions(self, function_calls: List[Dict]) -> List[Dict]:
        """Execute the detected function calls and return results"""
        results = []
        
        for call in function_calls:
            func_name = call["function"]
            parameters = call["parameters"]
            
            if func_name in self.available_functions:
                try:
                    function_to_call = self.available_functions[func_name]["function"]
                    result = function_to_call(**parameters)
                    
                    results.append({
                        "function": func_name,
                        "parameters": parameters,
                        "result": result,
                        "success": True
                    })
                    
                    print(f"✅ Successfully executed {func_name}")
                    
                except Exception as e:
                    results.append({
                        "function": func_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False
                    })
                    
                    print(f"❌ Error executing {func_name}: {e}")
            else:
                results.append({
                    "function": func_name,
                    "parameters": parameters,
                    "error": f"Function {func_name} not available",
                    "success": False
                })
        
        return results


class SangamGPTAgent:
    """
    Main agent class that orchestrates the function calling workflow.
    This represents the core AI assistant logic.
    """
    
    def __init__(self):
        self.function_handler = FunctionCallHandler()
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def process_query(self, user_query: str) -> str:
        """
        Main method to process user queries with function calling capability.
        
        Args:
            user_query: The user's question or request
            
        Returns:
            Final AI response incorporating function call results
        """
        print(f"\n💬 User Query: {user_query}")
        print("=" * 60)
        
        # Step 1: Create initial prompt with function descriptions
        functions_desc = self.function_handler.get_function_descriptions()
        
        initial_prompt = f"""You are SangamGPT, an expert historical AI assistant with access to a curated historical database.

Available functions you can call:
{functions_desc}

When you need specific data that might be in our database, call the appropriate function using this format:
- get_ruler_info("ruler name")
- get_battle_info("battle name") 
- get_timeline_events(start_year, end_year)

User query: {user_query}

If you need to call functions to answer this query accurately, include the function calls in your response. Otherwise, provide the best answer you can with your existing knowledge.
"""

        try:
            # Step 2: Get initial AI response
            print("🤖 AI is analyzing the query...")
            response = self.model.generate_content(initial_prompt)
            ai_response = response.text.strip()
            
            print("📝 Initial AI Response:")
            print(ai_response)
            print("-" * 40)
            
            # Step 3: Check for function calls
            function_calls = self.function_handler.detect_function_calls(ai_response)
            
            if not function_calls:
                print("ℹ️ No function calls detected - using direct AI response")
                return ai_response
            
            # Step 4: Execute function calls
            print(f"🔧 Executing {len(function_calls)} function call(s)...")
            function_results = self.function_handler.execute_functions(function_calls)
            
            # Step 5: Generate final response with function results
            results_summary = []
            for result in function_results:
                if result["success"]:
                    results_summary.append(f"Function {result['function']}: Success")
                else:
                    results_summary.append(f"Function {result['function']}: Error - {result['error']}")
            
            final_prompt = f"""Based on the function call results below, provide a comprehensive and well-formatted answer to the user's query: "{user_query}"

Function Call Results:
{json.dumps(function_results, indent=2)}

Please synthesize this information into a clear, informative, and engaging response. Use the actual data returned from the functions to provide accurate historical information.
"""
            
            print("🔄 Generating final response with function data...")
            final_response = self.model.generate_content(final_prompt)
            
            return final_response.text.strip()
            
        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error while processing your query: {str(e)}"
            print(error_msg)
            return error_msg


def main():
    """
    Main demonstration function showing various query types.
    """
    print("🏛️ Welcome to SangamGPT Function Calling Demo!")
    print("=" * 60)
    
    # Initialize our AI agent
    agent = SangamGPTAgent()
    
    # Test queries that demonstrate different function calling scenarios
    test_queries = [
        "Tell me about Ashoka - when did he rule and what was he known for?",
        "What happened in the Kalinga War and why was it significant?",
        "Show me what historical events occurred between 300 BCE and 300 CE",
        "Compare the reigns of Akbar and Napoleon - who ruled longer?",
        "Tell me about the Battle of Hastings",  # This should fail gracefully
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'🔍 EXAMPLE ' + str(i):=^60}")
        
        result = agent.process_query(query)
        
        print(f"\n✨ Final Response:")
        print(result)
        
        print("=" * 60)
        
        # Add a small delay between queries for readability
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()
