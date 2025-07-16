"""
Local LLM wrapper using LM Studio Python SDK.
"""
import os
import json
from dotenv import load_dotenv

# Load local environment (if needed for LM Studio)
load_dotenv()

try:
    import lmstudio as lms
except ImportError:
    raise ImportError(
        "LM Studio SDK not installed. Please run `pip install lmstudio`."
    )

class LocalLLMModelInterface:
    """Wrapper for local models via LM Studio."""
    def __init__(self, model_name: str):
        """Load or initialize the local model."""
        self.model_name = model_name
        # Load model into memory
        try:
            self.client = lms.llm(model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading local model '{model_name}': {e}")

    def generate(
        self,
        prompt: str,
        system_instruction: str = None,
        conversation_history: list = None
    ) -> str:
        """Generate text completion from local model."""
        # LM Studio responds to prompt only
        try:
            response = self.client.respond(prompt)
            return response
        except Exception as e:
            print(f"Error in local model generation: {e}")
            return ""

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = None,
        default_value: any = None
    ) -> any:
        """Generate and parse JSON from local model."""
        text = self.generate(prompt, system_instruction)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON from local model output.\nRaw: {text}")
            return default_value
