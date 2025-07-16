"""
LLM model wrapper for Gemini and Gemma models.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import time
import random
import json
from typing import List, Dict, Any, Optional, Union, Generator

from google import genai
from google.genai import types

from src.utils.config import get_model_config, ModelConfig

# Initialize Google GenAI client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class LLMModelInterface:
    """Interface for interacting with Google's LLM models."""
    
    def __init__(self, model_name: str):
        """Initialize the model interface.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.config = get_model_config(model_name)
        self.last_call_time = 0
        
    def _wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        # Calculate seconds per request based on rate limit
        seconds_per_request = 60 / self.config.rate_limit_rpm
        
        # Calculate time to wait
        elapsed = time.time() - self.last_call_time
        if elapsed < seconds_per_request:
            wait_time = seconds_per_request - elapsed
            time.sleep(wait_time)
        
        # Update last call time
        self.last_call_time = time.time()
    
    def generate(self, 
                prompt: str, 
                system_instruction: Optional[str] = None,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate content using specified model.
        
        Args:
            prompt: User prompt text
            system_instruction: Optional system instruction
            conversation_history: Optional conversation history
            
        Returns:
            Generated text response
        """
        self._wait_for_rate_limit()
        
        # Build contents with history if provided
        if conversation_history:
            contents = []
            for msg in conversation_history:
                role = msg.get("role", "user")
                text = msg.get("text", "")
                
                if role in ["user", "model"]:
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=text)]
                        )
                    )
            
            # Add current prompt
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            )
        else:
            # Just use the prompt
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
        
        # Configure generation
        if system_instruction and self.config.supports_system_instruction:
            config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                system_instruction=[types.Part.from_text(text=system_instruction)]
            )
        else:
            config = types.GenerateContentConfig(response_mime_type="text/plain")
        
        # Generate content
        output = ""
        try:
            for chunk in client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            ):
                output += chunk.text
        except Exception as e:
            print(f"Error generating content with {self.model_name}: {e}")
            # Return a basic error message in the same format
            output = f"Error: {str(e)}"
        
        return output
    
    def generate_json(self, 
                     prompt: str, 
                     system_instruction: Optional[str] = None,
                     default_value: Any = None) -> Any:
        """Generate JSON content and parse it.
        
        Args:
            prompt: User prompt text
            system_instruction: Optional system instruction for JSON format
            default_value: Default value to return if JSON parsing fails
            
        Returns:
            Parsed JSON object or default_value on failure
        """
        # Add JSON formatting instruction if not in the system instruction
        if system_instruction and "JSON" not in system_instruction:
            json_instruction = f"{system_instruction}\nFormat your response as valid JSON."
        elif not system_instruction:
            json_instruction = "Format your response as valid JSON."
        else:
            json_instruction = system_instruction
        
        # Generate response
        response = self.generate(prompt, json_instruction)
        
        # Try to extract JSON from the response
        try:
            # Check if the response contains JSON blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response.strip()
            
            # Parse the JSON
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response}")
            return default_value
