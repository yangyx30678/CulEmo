from inference_utils import process_file, write_json
import os
import json
import urllib.request

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

MODELS = ["gpt-5.6-sol"]
CONFIGS = [
    ("standard", 1),
    ("standard", 20),
    ("conceptual_chaining", 1),
    ("conceptual_chaining", 20)
]

# ponytail: Load env manually to avoid python-dotenv dependency
def load_dotenv():
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")


# ponytail: Direct API calls via urllib instead of installing 'openai' dependency
def get_prediction(model: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the GPT model.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            prediction = res_data["choices"][0]["message"]["content"]
            return prompt, prediction
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")


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
