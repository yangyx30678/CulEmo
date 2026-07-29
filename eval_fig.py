import json
import os
import argparse

# ponytail: Optional imports to allow running on minimal Python environments without dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from qa_metrics.em import em_match
except ImportError:
    # ponytail: Simple exact match fallback
    def em_match(gt_list, pred):
        pred_clean = str(pred).strip().lower()
        return any(str(gt).strip().lower() == pred_clean for gt in gt_list)

try:
    from sklearn.metrics import accuracy_score
except ImportError:
    # ponytail: Simple accuracy fallback
    def accuracy_score(y_true, y_pred):
        if not y_true:
            return 0.0
        return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def evaluate_predictions_binary(json_path: str) -> float:
    if not os.path.exists(json_path):
        return 0.0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return 0.0

    y_true_binary = []
    y_pred_binary = []

    for item in data:
        ground_truth = item.get("emotion", "")
        prediction = item.get("pred_emotion", "")
        match = em_match([ground_truth], prediction)

        y_true_binary.append(1)
        y_pred_binary.append(1 if match else 0)

    if len(y_true_binary) == 0:
        return 0.0

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple model predictions and generate comparative charts.")
    parser.add_argument(
        "--outputs_dir", 
        type=str, 
        default="outputs", 
        help="Directory containing the model output folders"
    )
    args = parser.parse_args()
    base_dir = args.outputs_dir

    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return

    LANGUAGE_MAP = {
        "English": "eng",
        "Arabic": "ara",
        "Spanish": "spn",
        "German": "deu",
        "Amharic": "amh",
        "Hindi": "hin"
    }

    COUNTRY_MAP = {
        "Ethiopia": "eth-eng",
        "UAE": "uae-eng",
        "Germany": "deu-eng",
        "India": "ind-eng",
        "Mexico": "mex-eng"
    }

    def find_file_by_prefix(directory, prefix):
        if not os.path.exists(directory):
            return None
        for f in os.listdir(directory):
            if f.startswith(prefix) and f.endswith(".json"):
                return os.path.join(directory, f)
        return None

    models = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    models.sort()
    
    if not models:
        print("No models found in the outputs directory.")
        return

    all_lang_results = {model: [] for model in models}
    all_country_results = {model: [] for model in models}
    
    languages = list(LANGUAGE_MAP.keys())
    countries = list(COUNTRY_MAP.keys())

    print("=== Scanning Models ===")
    for model in models:
        print(f"Processing model: {model}")
        model_dir = os.path.join(base_dir, model)
        lang_dir = os.path.join(model_dir, "lang")
        country_dir = os.path.join(model_dir, "countries")

        for lang, prefix in LANGUAGE_MAP.items():
            file_path = find_file_by_prefix(lang_dir, prefix)
            acc = evaluate_predictions_binary(file_path) if file_path else 0.0
            all_lang_results[model].append(acc)

        for country, prefix in COUNTRY_MAP.items():
            file_path = find_file_by_prefix(country_dir, prefix)
            acc = evaluate_predictions_binary(file_path) if file_path else 0.0
            all_country_results[model].append(acc)

    if HAS_PLOT:
        # Plotting grouped bar charts
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        x_lang = np.arange(len(languages))
        x_country = np.arange(len(countries))
        width = 0.8 / len(models) # dynamic width based on number of models

        # Draw Language Chart
        for i, model in enumerate(models):
            offset = i * width - (len(models) * width) / 2 + width / 2
            ax1.bar(x_lang + offset, all_lang_results[model], width, label=model)

        ax1.set_title("Comprehensive Language Accuracy Comparison")
        ax1.set_ylabel("Accuracy")
        ax1.set_xticks(x_lang)
        ax1.set_xticklabels(languages)
        ax1.set_ylim(0, 1.05)
        ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        # Draw Country Chart
        for i, model in enumerate(models):
            offset = i * width - (len(models) * width) / 2 + width / 2
            ax2.bar(x_country + offset, all_country_results[model], width, label=model)

        ax2.set_title("Comprehensive Country Accuracy Comparison")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(x_country)
        ax2.set_xticklabels(countries)
        ax2.set_ylim(0, 1.05)
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout()
        output_fig = os.path.join(base_dir, "comprehensive_evaluation_results.png")
        plt.savefig(output_fig, bbox_inches='tight')
        print(f"\nComprehensive figure saved to {output_fig}")
    else:
        # ponytail: Gracefully report missing plot library instead of failing
        print("\nNote: matplotlib/numpy not installed. Skipping chart generation.")

    # ==========================================
    # SAVE AND PRINT NUMERICAL SCORES
    # ==========================================
    print("\n" + "="*50)
    print("LANGUAGE ACCURACY SCORES")
    print("="*50)
    if HAS_PANDAS:
        df_lang = pd.DataFrame(all_lang_results, index=languages)
        print(df_lang.round(4).to_string())
    else:
        # ponytail: Fallback pretty printing without pandas
        print(f"{'Language':<15} " + " ".join(f"{m:>15}" for m in models))
        for idx, lang in enumerate(languages):
            row_vals = [all_lang_results[m][idx] for m in models]
            print(f"{lang:<15} " + " ".join(f"{v:>15.4f}" for v in row_vals))
    
    print("\n" + "="*50)
    print("COUNTRY ACCURACY SCORES")
    print("="*50)
    if HAS_PANDAS:
        df_country = pd.DataFrame(all_country_results, index=countries)
        print(df_country.round(4).to_string())
    else:
        # ponytail: Fallback pretty printing without pandas
        print(f"{'Country':<15} " + " ".join(f"{m:>15}" for m in models))
        for idx, country in enumerate(countries):
            row_vals = [all_country_results[m][idx] for m in models]
            print(f"{country:<15} " + " ".join(f"{v:>15.4f}" for v in row_vals))
    
    # Save to CSV
    output_csv_lang = os.path.join(base_dir, "comprehensive_lang_results.csv")
    output_csv_country = os.path.join(base_dir, "comprehensive_country_results.csv")
    
    if HAS_PANDAS:
        df_lang.to_csv(output_csv_lang)
        df_country.to_csv(output_csv_country)
    else:
        # ponytail: Fallback CSV writing without pandas
        import csv
        with open(output_csv_lang, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Language"] + models)
            for idx, lang in enumerate(languages):
                writer.writerow([lang] + [all_lang_results[m][idx] for m in models])
                
        with open(output_csv_country, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Country"] + models)
            for idx, country in enumerate(countries):
                writer.writerow([country] + [all_country_results[m][idx] for m in models])
    
    print(f"\nNumerical results saved to:")
    print(f"- {output_csv_lang}")
    print(f"- {output_csv_country}")

if __name__ == "__main__":
    main()
