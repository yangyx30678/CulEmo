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

MODELS = ["gemma4:e4b", "llama3.1:8b"]
CONFIGS = [
    ("standard", 1),
    ("standard", 20),
    ("conceptual_chaining", 1),
    ("conceptual_chaining", 20)
]

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


