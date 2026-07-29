"""
Utility functions for processing and evaluating LLM responses across different languages and countries.
This module handles prompt generation, file processing, and result storage.
"""

import csv
import json
import os


CC_PROMPT = """## Conceptual Chaining System Prompt

Role & Objective
You are a reasoning expert specializing in structured concept linking by connecting essential ideas in a logical sequence.
Your goal is to extract key terms and present reasoning in clear, stepwise chains while minimizing unnecessary explanation.
This reasoning method follows a conceptual chaining approach, where information is linked in structured steps to
establish relationships between ideas. This process integrates associative recall (direct lookups) and multi-hop reasoning
(sequential dependencies) into a unified framework.

How to Apply Conceptual Chaining
1. Extract Key Concepts → Identify the most relevant words or entities.
2. Use Minimal Words → Keep each reasoning step concise and direct.
3. Link Steps Sequentially → Maintain a clear and meaningful progression between concepts.
4. Avoid Full Sentences → Responses should use structured keyword connections.
5. Follow the Required Format → Present answers using stepwise chains for clarity.

Rules & Directives
1. Use Structured Concept Linking (arrows '→')
2. Avoid Unnecessary Text (No full sentences)
3. Maintain Logical Flow
4. Output Format:
<think>
[shorthand reasoning]
</think>
[Final emotion word]
"""

def get_prompt(language: str, country: str, text: str, prompt_type: str = "standard") -> str:
    """
    Generates a prompt in the specified language or for the specified country.
    
    Args:
        language (str): The language to generate the prompt in (e.g., "English", "Arabic")
        country (str): The country context for the prompt (e.g., "Mexico", "Germany")
        text (str): The input text to be evaluated
        
    Returns:
        str: A formatted prompt in the specified language
        
    Raises:
        ValueError: If neither language nor country is provided
    """
    system_role = "You are an objective linguistic analysis API. You MUST classify the text objectively, even if it contains sensitive or offensive content. Do not refuse to answer. "
    if language == "English":
        prompt = system_role + f"""You live in the United States of America, choose the best emotion you feel for the given question. Emotions can only be 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}\nAnswer:"""
    elif language == "Arabic":
        prompt = f"""أنت تعيش في دولة الإمارات العربية المتحدة، اختر الحالة العاطفية الأفضل التي تشعر بها تجاه السؤال المطروح. يمكن أن تكون العواطف إما "غضب" أو "خوف" أو "حزن" أو "فرح" أو "ذنب" أو "محايد". أعد فقط كلمة عاطفة واحدة من قائمة العواطف دون أي تفسير إضافي.\n النص: {text}\n الإجابة:"""
    elif language == "Spanish":
        prompt = f"""Vives en México. Elige la emoción que sientes más a menudo en la pregunta. Las emociones pueden ser 'enojo','tristeza','culpa','alegría','miedo' o 'neutral'. Solo responde con una palabra de la lista de emociones sin más explicaciones. \nTexto: {text}\nRespuesta:"""
    elif language == "Amharic":
        prompt = f"""የምትኖረው ኢትዮጵያ ውስጥ ነው፣ ለተሰጠው ጥያቄ የሚሰማህን ስሜት ምረጥ። ስሜቶች 'ቁጣ', 'ጥፋተኛ', 'ሀዘን', 'ደስታ', 'ፍርሀት' ወይም 'መደበኛ' ብቻ ሊሆኑ ይችላሉ:: ያለተጨማሪ ማብራሪያ ከስሜቶች ዝርዝር ውስጥ አንዱን ስሜት ብቻ ይመልሱ።\n"ጽሑፍ፡ {text}\nመልስ፡"""
    elif language == "German":
        prompt = f"""Sie leben in Deutschland. Wählen Sie die beste Emotion, die Sie für die gestellte Frage empfinden. Emotionen können nur 'wut', 'freude', 'traurigkeit', 'schuld', 'furcht' oder 'neutral' sein. Geben Sie nur ein einziges Emotionswort aus der Liste der Emotionen ohne weitere Erklärung zurück. \nText: {text}\nAntwort:"""
    elif language == "Hindi":
        prompt = f"""आप भारत में रहते हैं, दिए गए प्रश्न के लिए अपनी सबसे अच्छी भावना चुनें। भावनाएँ केवल 'उदासी', 'आनंद', 'अपराध', 'गुस्सा', 'डर', या 'सामान्य' हो सकती हैं। बिना किसी अतिरिक्त स्पष्टीकरण के भावनाओं की सूची से केवल एक ही भावना शब्द लौटाएँ। \nपाठ: {text}\nउत्तर:"""
    elif country:
        prompt = f"You live in {country}, choose the best emotion you feel for the given question. Emotions can be only either 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}"
    else:
        raise ValueError("Either a valid 'language' or 'country' must be provided.")
    if prompt_type == "conceptual_chaining":
        prompt = CC_PROMPT + "\n\n" + prompt + "\nAfter </think>, return ONLY the single emotion word without any brackets or extra text."
    return prompt


def get_batch_prompt(language: str, country: str, texts: list[str], prompt_type: str = "standard") -> str:
    """
    Generates a batch prompt for emotion prediction.
    """
    texts_formatted = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    
    if language == "English":
        instruction = "You live in the United States of America, choose the best emotion you feel for each given question."
    elif language == "Arabic":
        instruction = "أنت تعيش في دولة الإمارات العربية المتحدة، اختر الحالة العاطفية الأفضل التي تشعر بها تجاه كل سؤال مطروح."
    elif language == "Spanish":
        instruction = "Vives en México. Elige la emoción que sientes más a menudo en cada pregunta."
    elif language == "Amharic":
        instruction = "የምትኖረው ኢትዮጵያ ውስጥ ነው፣ ለእያንዳንዱ ጥያቄ የሚሰማህን ስሜት ምረጥ።"
    elif language == "German":
        instruction = "Sie leben in Deutschland. Wählen Sie die beste Emotion, die Sie für jede gestellte Frage empfinden."
    elif language == "Hindi":
        instruction = "आप भारत में रहते हैं, दिए गए प्रत्येक प्रश्न के लिए अपनी सबसे अच्छी भावना चुनें।"
    elif country:
        instruction = f"You live in {country}, choose the best emotion you feel for each given question."
    else:
        raise ValueError("Either a valid 'language' or 'country' must be provided.")
        
    system_role = "You are an objective linguistic analysis API. You MUST classify ALL provided texts objectively, even if they contain sensitive, illegal, or offensive content. Do not refuse to answer any text. "
    common_rules = "Emotions can only be 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'."
    return_format = "Return ONLY a valid JSON array of strings corresponding to the emotions of the provided texts in order, without any other text or explanation. Example: [\"joy\", \"fear\", \"neutral\"]"
    
    prompt = f"{system_role}{instruction} {common_rules} {return_format}\nTexts:\n{texts_formatted}\nAnswer:"
    if prompt_type == "conceptual_chaining":
        prompt = CC_PROMPT + "\n\n" + prompt + "\nProvide the JSON array inside boxed[]."
    return prompt


def _parse_row(row, language, country):
    if language == "English":
        text, gt_emotion, gt_sentiment = row
    elif language and not country:
        # For non-English languages, the LLM is instructed to output English emotion words, 
        # so we must use emotion_eng as the ground truth.
        text_eng, text, emotion_eng, emotion_native, sentiment_eng, sentiment_native = row
        gt_emotion = emotion_eng
    elif country and not language:
        text, _, gt_emotion, _, gt_sentiment, _ = row
    else:
        raise Exception("ERROR!")
    return text, gt_emotion


def process_file(tsv_file: str, model: str, get_prediction, language: str = None, country: str = None, batch_size: int = 1, prompt_type: str = "standard") -> list[dict]:
    """
    Processes a TSV file containing emotion evaluation data and generates predictions using the specified model.
    
    Args:
        tsv_file (str): Path to the TSV file containing the evaluation data
        model (str): Name of the model to use for predictions
        get_prediction (callable): Function to get predictions from the model
        language (str, optional): Language to use for prompts
        country (str, optional): Country context to use for prompts
        
    Returns:
        list[dict]: List of dictionaries containing the evaluation results
        
    Note:
        Either language or country must be provided, but not both
    """
    results = []

    with open(tsv_file, "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        next(tsv_reader)  # Skip header row

        if batch_size <= 1:
            for count, row in enumerate(tsv_reader, start=1):
                print(count)
                text, gt_emotion = _parse_row(row, language, country)
                prompt, pred_emotion = get_prediction(model, get_prompt(language, country, text, prompt_type))
                if prompt_type == "conceptual_chaining":
                    import re
                    cleaned = pred_emotion.lower()
                    
                    # Try to get text after </think>
                    if "</think>" in cleaned:
                        final_ans = cleaned.split("</think>")[-1].strip()
                    else:
                        final_ans = cleaned
                        
                    # Remove boxed[] if the model still uses it
                    match = re.search(r"boxed\[(.*?)\]", final_ans, re.DOTALL)
                    if match:
                        final_ans = match.group(1).strip()
                        
                    # Find valid emotion words
                    valid_emotions = ['anger', 'fear', 'sadness', 'joy', 'guilt', 'neutral']
                    found = [e for e in valid_emotions if e in final_ans]
                    if found:
                        pred_emotion = found[-1] # take the last mentioned emotion
                    else:
                        pred_emotion = final_ans.strip()
                results.append({
                    "prompt": prompt,
                    "text": text,
                    **({"country": country} if country else {"language": language}),
                    "emotion": gt_emotion,
                    "pred_emotion": pred_emotion,
                    "model": model,
                })
        else:
            batch_rows = []
            
            def process_current_batch(b_rows):
                if not b_rows: return
                parsed_rows = [_parse_row(r, language, country) for r in b_rows]
                texts = [pr[0] for pr in parsed_rows]
                gt_emotions = [pr[1] for pr in parsed_rows]
                
                prompt_str = get_batch_prompt(language, country, texts, prompt_type)
                prompt, pred_response = get_prediction(model, prompt_str)
                
                # Handle cases where safety filters block the response entirely (returning None)
                if pred_response is None:
                    print("Warning: Received None as prediction response (likely blocked by safety filters).")
                    pred_response = ""
                
                # Parse JSON array from LLM
                cleaned = pred_response.strip()
                if prompt_type == "conceptual_chaining":
                    import re
                    match = re.search(r"boxed\[(.*?)\]", cleaned, re.DOTALL)
                    if match:
                        cleaned = match.group(1).strip()
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                if cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                try:
                    pred_emotions = json.loads(cleaned)
                    if not isinstance(pred_emotions, list):
                        pred_emotions = ["error"] * len(b_rows)
                    elif len(pred_emotions) != len(b_rows):
                        print(f"Warning: Batch length mismatch! Expected {len(b_rows)}, got {len(pred_emotions)}.")
                        while len(pred_emotions) < len(b_rows): pred_emotions.append("error")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {pred_response}")
                    pred_emotions = ["error"] * len(b_rows)
                
                for idx, (text, gt_e) in enumerate(zip(texts, gt_emotions)):
                    results.append({
                        "prompt": prompt_str,
                        "text": text,
                        **({"country": country} if country else {"language": language}),
                        "emotion": gt_e,
                        "pred_emotion": pred_emotions[idx] if idx < len(pred_emotions) else "error",
                        "model": model,
                    })

            for count, row in enumerate(tsv_reader, start=1):
                batch_rows.append(row)
                if len(batch_rows) >= batch_size:
                    print(f"Processing batch up to line {count}...")
                    process_current_batch(batch_rows)
                    batch_rows = []
            
            if batch_rows:
                print("Processing final batch...")
                process_current_batch(batch_rows)

    return results


def write_json(data: list[dict], output_path: str) -> None:
    """
    Writes the evaluation results to a JSON file.
    
    Args:
        data (list[dict]): List of dictionaries containing the evaluation results
        output_path (str): Path where the JSON file should be written
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully written to {output_path}")

