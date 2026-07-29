"""
Script for running inference using Ollama models.
This script processes text data and generates emotion predictions using locally hosted Ollama models.
It supports both language-specific and country-specific evaluations.
"""

from inference_utils import process_file, write_json
from ollama import chat, ChatResponse

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

MODEL_NAME = "gemma4:e4b"  # Local Llama model to use (e.g. "llama3.2", "llama3.1:8b")
# MODEL_NAME = "llama3.1:8b"  # Local Llama model to use (e.g. "llama3.2", "llama3.1:8b")
# Fix invalid Windows filenames by replacing ':' with '_'
SAFE_MODEL_NAME = MODEL_NAME.replace(":", "_")
 
PROMPT_TYPE = "conceptual_chaining"  # "standard" or "conceptual_chaining"

OUTPUT_DIR_LANG = f"outputs/{SAFE_MODEL_NAME}_{PROMPT_TYPE}/lang"
OUTPUT_DIR_COUNTRY = f"outputs/{SAFE_MODEL_NAME}_{PROMPT_TYPE}/countries"
BATCH_SIZE = 20     # Batch size (set to 1 to bypass safety filters acting on entire batches, or >1 for speed)


def get_prediction(model: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the local Llama model using Ollama.
    
    Args:
        model (str): Name of the Llama model to use
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
    """
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response["message"]["content"])
    return prompt, response["message"]["content"]


if __name__ == "__main__":
    # 1. Run Language Evaluations
    print("=== Starting Language Evaluations ===", MODEL_NAME)
    import os
    for lang, prefix in LANGUAGE_MAP.items():
        print(f"\n[Language: {lang}]")
        tsv_path = f"data/test/{prefix}.tsv"
        output_json = f"{OUTPUT_DIR_LANG}/{prefix}_{SAFE_MODEL_NAME}.json"
        
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
        output_json = f"{OUTPUT_DIR_COUNTRY}/{out_prefix}_{SAFE_MODEL_NAME}.json"
        
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


