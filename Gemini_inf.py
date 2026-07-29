"""
Script for running inference using Google's Gemini models.
This script processes text data and generates emotion predictions using Gemini models.
It supports both language-specific and country-specific evaluations.
"""
from google import genai
from inference_utils import process_file, write_json
import os
from dotenv import load_dotenv

# Configuration for the evaluation
LANGUAGE_MAP = {
    "English": "eng",
    "Arabic": "ara",
    "Spanish": "spn",
    "German": "deu",
    "Amharic": "amh",
    "Hindi": "hin"
}

# Country mapped to (TSV prefix, Output file prefix)
COUNTRY_MAP = {
    "Ethiopia": ("amh", "eth-eng"),
    "United Arab Emirates": ("ara", "uae-eng"),
    "Germany": ("deu", "deu-eng"),
    "India": ("hin", "ind-eng"),
    "Mexico": ("spn", "mex-eng")
}

MODEL_NAME = "gemini-3-flash-preview"
PROMPT_TYPE = "standard"  # or "conceptual_chaining"
OUTPUT_DIR_LANG = f"outputs/{MODEL_NAME}_{PROMPT_TYPE}/lang"
OUTPUT_DIR_COUNTRY = f"outputs/{MODEL_NAME}_{PROMPT_TYPE}/countries"
BATCH_SIZE = 20     # Set to > 1 to enable batch prediction, e.g., 20

load_dotenv()
# Get API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

# Initialize Gemini API
client = genai.Client(api_key=API_KEY)


import time
from google.genai import errors

def get_prediction(model_name: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the Gemini model.
    
    Args:
        model_name (str): Name of the Gemini model to use (e.g., "gemini-1.5-flash")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
        
    Note:
        The model is configured to return a single emotion word from the allowed set:
        'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'
    """
    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # 固定 RPM 為 5，也就是每次請求間隔至少 12 秒 (60秒 / 5)
            time.sleep(12)
            return prompt, response.text
        except errors.ClientError as e:
            if "429" in str(e):
                print("Rate limit reached (429). Sleeping for 15 seconds before retrying...")
                time.sleep(15)
            else:
                raise e


if __name__ == "__main__":
    # 1. Run Language Evaluations
    print("=== Starting Language Evaluations ===")
    import os
    for lang, prefix in LANGUAGE_MAP.items():
        print(f"\n[Language: {lang}]")
        tsv_path = f"data/test/{prefix}.tsv"
        output_json = f"{OUTPUT_DIR_LANG}/{prefix}_{MODEL_NAME}.json"
        
        if os.path.exists(output_json):
            print(f"-> {output_json} already exists. Skipping {lang}.")
            continue
            
        output_data = process_file(
            tsv_file=tsv_path,
            model=MODEL_NAME,
            get_prediction=get_prediction,
            language=lang,
            country=None,
            batch_size=BATCH_SIZE,
            prompt_type=PROMPT_TYPE
        )
        write_json(output_data, output_json)
        print(f"Finished {lang}!")

    # 2. Run Country Evaluations
    print("\n=== Starting Country Evaluations ===")
    for country, (prefix, out_prefix) in COUNTRY_MAP.items():
        print(f"\n[Country: {country}]")
        tsv_path = f"data/test/{prefix}.tsv"
        output_json = f"{OUTPUT_DIR_COUNTRY}/{out_prefix}_gemini3.1_flash.json"
        
        if os.path.exists(output_json):
            print(f"-> {output_json} already exists. Skipping {country}.")
            continue
            
        output_data = process_file(
            tsv_file=tsv_path,
            model=MODEL_NAME,
            get_prediction=get_prediction,
            language=None,
            country=country,
            batch_size=BATCH_SIZE,
            prompt_type=PROMPT_TYPE
        )
        write_json(output_data, output_json)
        print(f"Finished {country}!")


