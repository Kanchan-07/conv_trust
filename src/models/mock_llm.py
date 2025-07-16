"""
Mock LLM implementation for trust dataset generation.
"""
import os
import time
import random
import json
from typing import List, Dict, Any, Optional, Union

from src.utils.config import get_model_config, ModelConfig

class MockLLMModelInterface:
    """Mock interface for generating synthetic conversations without external API calls."""
    
    def __init__(self, model_name: str):
        """Initialize the mock model interface.
        
        Args:
            model_name: Name of the model to use (for compatibility)
        """
        self.model_name = model_name
        # Attempt to get config, but don't fail if model not found
        try:
            self.config = get_model_config(model_name)
        except ValueError:
            # Create a default config
            self.config = ModelConfig(name=model_name, rate_limit_rpm=10, daily_limit=1000)
        self.last_call_time = 0
        
        # Load example templates
        self.templates = self._load_templates()
        
    def _load_templates(self) -> List[Dict[str, Any]]:
        """Load example conversation templates."""
        # Simple example templates
        return [
            {
                "turns": [
                    {"turn_id": 1, "speaker": "user", "utterance": "Hello, I need help with my order. It's been a week and it hasn't arrived yet."},
                    {"turn_id": 2, "speaker": "agent", "utterance": "I'm sorry to hear that. I'd be happy to help you track down your order. Could you please provide your order number?"},
                    {"turn_id": 3, "speaker": "user", "utterance": "Sure, it's #12345678."},
                    {"turn_id": 4, "speaker": "agent", "utterance": "Thank you. Let me check that for you... I can see that your order was shipped two days ago and is currently in transit. According to the tracking information, it should arrive by tomorrow."},
                    {"turn_id": 5, "speaker": "user", "utterance": "That's a relief. I was getting worried because the estimated delivery date was yesterday."},
                    {"turn_id": 6, "speaker": "agent", "utterance": "I understand your concern. There seems to have been a slight delay with the carrier, but it's on its way now. Would you like me to send you the tracking link so you can monitor it?"}
                ]
            },
            {
                "turns": [
                    {"turn_id": 1, "speaker": "user", "utterance": "Hi, I'm looking for a new laptop for graphic design work. Can you recommend something?"},
                    {"turn_id": 2, "speaker": "agent", "utterance": "I'd be happy to help you find a suitable laptop for graphic design. What's your budget range?"},
                    {"turn_id": 3, "speaker": "user", "utterance": "I'm thinking around $1,500, but I can stretch to $2,000 if it's really worth it."},
                    {"turn_id": 4, "speaker": "agent", "utterance": "Great! For graphic design, you'll want a laptop with a good display, powerful processor, and dedicated graphics card. The MacBook Pro 14-inch and Dell XPS 15 are excellent options in that price range."},
                    {"turn_id": 5, "speaker": "user", "utterance": "I'm more familiar with Windows, so the Dell sounds interesting. What specs does it have?"},
                    {"turn_id": 6, "speaker": "agent", "utterance": "The Dell XPS 15 comes with an 11th Gen Intel i7 processor, 16GB of RAM, 512GB SSD, and an NVIDIA GTX 3050 Ti graphics card. It has a beautiful 15.6-inch 3.5K OLED display with 100% DCI-P3 color gamut, which is ideal for graphic design work."}
                ]
            },
            {
                "turns": [
                    {"turn_id": 1, "speaker": "user", "utterance": "I need to book a flight from New York to London next month."},
                    {"turn_id": 2, "speaker": "agent", "utterance": "I'd be happy to help you book a flight. Could you please provide your preferred travel dates?"},
                    {"turn_id": 3, "speaker": "user", "utterance": "I want to leave on June 15th and return on June 25th."},
                    {"turn_id": 4, "speaker": "agent", "utterance": "Thank you. Are you traveling alone or with others? And do you have a preference for airlines or class of travel?"},
                    {"turn_id": 5, "speaker": "user", "utterance": "Just me. I'd prefer economy class, and I've had good experiences with British Airways in the past."},
                    {"turn_id": 6, "speaker": "agent", "utterance": "I've found several British Airways flights for those dates. There's a non-stop flight leaving JFK at 10:30 PM on June 15th, arriving at Heathrow at 10:30 AM the next day. For the return, there's a flight leaving Heathrow at 2:15 PM on June 25th. The total for the round trip in economy is $950. Would you like to proceed with this option?"}
                ]
            }
        ]
        
    def _wait_for_rate_limit(self):
        """Simulate wait for rate limits."""
        # Calculate seconds per request based on rate limit
        seconds_per_request = 60 / self.config.rate_limit_rpm
        
        # Calculate time to wait
        elapsed = time.time() - self.last_call_time
        if elapsed < seconds_per_request:
            wait_time = seconds_per_request - elapsed
            time.sleep(wait_time)
        
        # Update last call time
        self.last_call_time = time.time()
    
    def _adapt_template_to_scenario(self, template: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Adapt a template to a specific scenario by modifying utterances.
        
        Args:
            template: Conversation template
            scenario: Target scenario
            
        Returns:
            Adapted conversation
        """
        # Create a copy of the template
        adapted = {"turns": []}
        
        # Process each turn
        for turn in template["turns"]:
            new_turn = turn.copy()
            
            # Modify the initial user query to reflect the scenario
            if turn["turn_id"] == 1 and turn["speaker"] == "user":
                # Set a scenario-specific opening based on the scenario
                if "customer service" in scenario.lower():
                    new_turn["utterance"] = f"Hello, I need help with a {scenario.lower()} issue."
                elif "recommendation" in scenario.lower():
                    new_turn["utterance"] = f"Hi, can you help me with {scenario.lower()}?"
                elif "booking" in scenario.lower() or "scheduling" in scenario.lower():
                    new_turn["utterance"] = f"I'd like to {scenario.lower()}. Can you assist me?"
                else:
                    # Keep original but mention scenario
                    new_turn["utterance"] = f"Hello, I need help with {scenario.lower()}."
            
            adapted["turns"].append(new_turn)
            
        return adapted
    
    def generate(self, 
                prompt: str, 
                system_instruction: Optional[str] = None,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate mock content.
        
        Args:
            prompt: User prompt text
            system_instruction: Optional system instruction
            conversation_history: Optional conversation history
            
        Returns:
            Generated text response
        """
        self._wait_for_rate_limit()
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 2.0))
        
        # Return a simple response based on the prompt
        if "error" in prompt.lower():
            return "I apologize, but I encountered an error processing your request."
        else:
            return "I've processed your request and generated an appropriate response."
    
    def generate_json(self, 
                     prompt: str, 
                     system_instruction: Optional[str] = None,
                     default_value: Any = None) -> Any:
        """Generate mock JSON content.
        
        Args:
            prompt: User prompt text
            system_instruction: Optional system instruction for JSON format
            default_value: Default value to return if JSON parsing fails
            
        Returns:
            JSON object
        """
        self._wait_for_rate_limit()
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 2.0))
        
        # Check if this is a conversation generation prompt
        if "conversation" in prompt.lower() and "scenario" in prompt.lower():
            # Extract the scenario from the prompt
            scenario_start = prompt.find("scenario:")
            scenario_end = prompt.find(".", scenario_start)
            
            if scenario_start != -1 and scenario_end != -1:
                scenario = prompt[scenario_start + 9:scenario_end].strip()
            else:
                scenario = "general customer service inquiry"
                
            # Select a random template and adapt it to the scenario
            template = random.choice(self.templates)
            conversation = self._adapt_template_to_scenario(template, scenario)
            
            return conversation
        else:
            # Return default value or a basic structure
            if default_value:
                return default_value
            else:
                return {"result": "This is a mock response", "status": "success"}
