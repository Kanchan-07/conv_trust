#!/usr/bin/env python
"""
Trust Research Dataset Generator

This script generates a synthetic dataset for trust research in conversational agents.
It creates conversation data with annotated trust metrics at both turn and conversation levels.
"""
import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import DatasetConfig, GEMINI_MODELS, ensure_output_dirs
from src.generators.dataset import DatasetGenerator
from src.generators.conversation import ConversationGenerator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate trust research dataset")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num-conversations", 
        type=int, 
        default=20,
        help="Number of conversations to generate"
    )
    parser.add_argument(
        "--min-turns", 
        type=int, 
        default=4,
        help="Minimum turns per conversation"
    )
    parser.add_argument(
        "--max-turns", 
        type=int, 
        default=10,
        help="Maximum turns per conversation"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        nargs="+", 
        default=["gemini-2.0-flash-lite"],
        help="Models to use for generation"
    )
    return parser.parse_args()

def validate_env():
    """Validate environment variables."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.")
        print("Please set this variable in your .env file.")
        return False
    return True

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    if not validate_env():
        return
    
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = DatasetConfig(
        output_dir=args.output_dir,
        num_conversations=args.num_conversations,
        conversation_turns_range=(args.min_turns, args.max_turns)
    )
    
    # Validate models
    valid_models = []
    for model in args.models:
        if model in GEMINI_MODELS:
            valid_models.append(model)
        else:
            print(f"Warning: Unknown model '{model}', skipping.")
    
    if not valid_models:
        print(f"Error: No valid models specified. Available models:")
        for model, cfg in GEMINI_MODELS.items():
            print(f"  - {model} (rate limit: {cfg.rate_limit_rpm} rpm, daily limit: {cfg.daily_limit})")
        return
    
    # Create generator
    generator = DatasetGenerator(config)
    
    # Start time
    start_time = time.time()
    print(f"Starting dataset generation with {len(valid_models)} models")
    print(f"Target: {args.num_conversations} conversations")
    
    try:
        # Generate dataset
        stats = generator.generate_dataset(valid_models)
        
        # End time
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        print("\nDataset generation complete!")
        print(f"Generated {len(stats['filepaths'])} conversations")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output directory: {os.path.abspath(config.output_dir)}")
        
        # Print models used
        print("\nModels used:")
        for model, count in stats["conversations_per_model"].items():
            print(f"  - {model}: {count} conversations")
        
        # Print dataset info file path
        info_path = os.path.join(config.output_dir, "dataset_info.json")
        print(f"\nDataset info saved to: {os.path.abspath(info_path)}")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")

def generate_single_example():
    """Generate a single example conversation and print to console."""
    # Load environment variables
    load_dotenv()
    
    # Validate environment
    if not validate_env():
        return
    
    print("Generating a single example conversation...")
    generator = ConversationGenerator("gemini-2.0-flash")
    conversation = generator.generate_conversation(
        scenario="Customer service interaction about a delayed package",
        user_id="example_user",
        num_turns=4
    )
    
    print("\nExample conversation:")
    print(json.dumps(conversation.to_dict(), indent=2))

if __name__ == "__main__":
    # If no arguments provided, generate a single example
    if len(sys.argv) == 1:
        main()
    else:
        main()
