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

# MODELS = ["gemini-2.5-flash"] # Using a standard available model name since 3-flash is preview or you can keep gemini-3-flash-preview
# Let's keep the one that was there originally
MODELS = ["gemini-3.1-flash-lite"]
CONFIGS = [
    # ("standard", 1),
    ("standard", 20),
    # ("conceptual_chaining", 1),
    ("conceptual_chaining", 20)
]

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
    """
    while True:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            # 固定 RPM 為 5，也就是每次請求間隔至少 12 秒 (60秒 / 5)
            time.sleep(5)
            return prompt, response.text
        except errors.ClientError as e:
            if "429" in str(e):
                print("Rate limit reached (429). Sleeping for 15 seconds before retrying...")
                time.sleep(15)
            else:
                raise e


if __name__ == "__main__":
    import os
    
    for model_name in MODELS:
        safe_model_name = model_name.replace(":", "_")
        for prompt_type, batch_size in CONFIGS:
            pt_label = "CC" if prompt_type == "conceptual_chaining" else prompt_type
            
            output_dir_lang = f"outputs/{safe_model_name}_{pt_label}_b{batch_size}/lang"
            output_dir_country = f"outputs/{safe_model_name}_{pt_label}_b{batch_size}/countries"
            
            print(f"\n{'='*50}")
            print(f"Running Model: {model_name} | Prompt: {prompt_type} | Batch: {batch_size}")
            print(f"{'='*50}")
            
            # 1. Run Language Evaluations
            for lang, prefix in LANGUAGE_MAP.items():
                print(f"\n[Language: {lang}]")
                tsv_path = f"data/test/{prefix}.tsv"
                output_json = f"{output_dir_lang}/{prefix}_{safe_model_name}.json"
                
                if os.path.exists(output_json):
                    print(f"-> {output_json} already exists. Skipping {lang}.")
                    continue
                    
                output_data = process_file(
                    tsv_file=tsv_path,
                    model=model_name,
                    get_prediction=get_prediction,
                    language=lang,
                    country=None,
                    batch_size=batch_size,
                    prompt_type=prompt_type
                )
                write_json(output_data, output_json)
                print(f"Finished {lang}!")

            # 2. Run Country Evaluations
            for country, (prefix, out_prefix) in COUNTRY_MAP.items():
                print(f"\n[Country: {country}]")
                tsv_path = f"data/test/{prefix}.tsv"
                output_json = f"{output_dir_country}/{out_prefix}_{safe_model_name}.json"
                
                if os.path.exists(output_json):
                    print(f"-> {output_json} already exists. Skipping {country}.")
                    continue
                    
                output_data = process_file(
                    tsv_file=tsv_path,
                    model=model_name,
                    get_prediction=get_prediction,
                    language=None,
                    country=country,
                    batch_size=batch_size,
                    prompt_type=prompt_type
                )
                write_json(output_data, output_json)
                print(f"Finished {country}!")


