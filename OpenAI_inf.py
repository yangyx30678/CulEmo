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

MODEL_NAME = "gpt-5.6-sol"
# Fix invalid Windows filenames by replacing ':' with '_' (if any)
SAFE_MODEL_NAME = MODEL_NAME.replace(":", "_")

OUTPUT_DIR_LANG = f"outputs/{SAFE_MODEL_NAME}/lang"
OUTPUT_DIR_COUNTRY = f"outputs/{SAFE_MODEL_NAME}/countries"
BATCH_SIZE = 20     # Set to > 1 to enable batch prediction, e.g., 20


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
    
    Args:
        model (str): Name of the GPT model to use (e.g., "gpt-4")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
        
    Note:
        The model is configured to return a single emotion word from the allowed set:
        'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'
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
    # 1. Run Language Evaluations
    print("=== Starting Language Evaluations ===", MODEL_NAME)
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
            batch_size=BATCH_SIZE
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
            batch_size=BATCH_SIZE
        )
        write_json(output_data, output_json)
        print(f"Finished {country}!")
