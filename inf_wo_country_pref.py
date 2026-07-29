"""
Script for running inference without country preferences.
This script processes text data and generates emotion predictions using various models.
It supports both language-specific and country-specific evaluations.
"""

import csv
from inference_utils import write_json
# ponytail: Commented out unused openai import to avoid unnecessary dependency error
# from openai import OpenAI
import os


# Configuration for the evaluation
TSV_FILE_PATH = "data/test/eng.tsv"
# Either LANG or COUNTRY must be set to None
LANG = "English"     # "English", "Arabic", "Spanish", "German", "Amharic", "Hindi", None
COUNTRY = None
MODEL_NAME = "gemini-1.5-flash"
PROMPT_TYPE = "standard"  # or "conceptual_chaining"
OUTPUT_JSON_PATH = f"wo_country_pref_outputs/wo_country_pref_gemini1.5_flash_{PROMPT_TYPE}/eng_wo-country-pref_gemini1.5_flash.json"


### Gemini 1.5 Flash
import google.generativeai as genai
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")
genai.configure(api_key=API_KEY)

def get_prediction(model_name: str, prompt):
    """
    Gets a prediction from the Gemini model.
    
    Args:
        model_name (str): Name of the Gemini model to use (e.g., "gemini-1.5-flash")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return prompt, response.text


#### Claude 3.5 Sonnet && Claude 3 Opus
# from anthropic import Anthropic
# API_KEY = os.getenv("ANTHROPIC_API_KEY")
# if not API_KEY:
#     raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
# client = Anthropic(api_key=API_KEY)
# def get_prediction(model: str, prompt):
#     response = client.messages.create(
#         model=model,
#         max_tokens=1000,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     prediction = response.content[0].text
#     return prompt, prediction


#### GPT-4
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("Please set the OPENAI_API_KEY environment variable")
# def get_prediction(model: str, prompt):
#         completion = client.chat.completions.create(
#             model=model,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         prediction = completion.choices[0].message.content
#         return prompt, prediction


CC_PROMPT = """## Conceptual Chaining System Prompt

Role & Objective
You are a reasoning expert specializing in structured concept linking by connecting essential ideas in a logical sequence.
Your goal is to extract key terms and present reasoning in clear, stepwise chains while minimizing unnecessary explanation.
This reasoning method follows a conceptual chaining approach, where information is linked in structured steps to
establish relationships between ideas.

How to Apply Conceptual Chaining
1. Extract Key Concepts → Identify the most relevant words or entities.
2. Use Minimal Words → Keep each reasoning step concise and direct.
3. Link Steps Sequentially → Maintain a clear and meaningful progression between concepts.
4. Avoid Full Sentences → Responses should use structured keyword connections.
5. Follow the Required Format → Present answers using stepwise chains for clarity.

Rules & Directives
1. Use Structured Concept Linking (arrows '→')
2. Avoid Unnecessary Text
3. Maintain Logical Flow
4. Output Format:
<think>
[shorthand reasoning]
</think>
[Final emotion word]
"""

def get_prompt(language, country, text, prompt_type="standard"):
    """
    Generates a prompt for emotion prediction based on language or country context.
    
    Args:
        language (str): The language to use for the prompt
        country (str): The country context (if any)
        text (str): The input text to analyze
        
    Returns:
        str: The formatted prompt
        
    Raises:
        ValueError: If neither language nor country is provided
    """
    if language == "English":
        prompt = f"""Choose the best emotion you feel for the given question. Emotions can be only either 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}\nAnswer:"""
    elif language == "Arabic":
        prompt = f"""اختر الحالة العاطفية الأفضل التي تشعر بها تجاه السؤال المطروح. يمكن أن تكون العواطف إما "غضب" أو "خوف" أو "حزن" أو "فرح" أو "ذنب" أو "محايد". أعد فقط كلمة عاطفة واحدة من قائمة العواطف دون أي تفسير إضافي.\n النص: {text}\n الإجابة:"""
    elif language == "Spanish":
        prompt = f"""Elige la emoción que sientes más a menudo en la pregunta. Las emociones pueden ser 'enojo','tristeza','culpa','alegría','miedo' o 'neutral'. Solo responde con una palabra de la lista de emociones sin más explicaciones. \nTexto: {text}\nRespuesta:"""
    elif language == "Amharic":
        prompt = f"""ለተሰጠው ጥያቄ የሚሰማህን ስሜት ምረጥ። ስሜቶች 'ቁጣ', 'ጥፋተኛ', 'ሀዘን', 'ደስታ', 'ፍርሀት' ወይም 'መደበኛ' ብቻ ሊሆኑ ይችላሉ:: ያለተጨማሪ ማብራሪያ ከስሜቶች ዝርዝር ውስጥ አንዱን ስሜት ብቻ ይመልሱ።\n"ጽሑፍ፡ {text}\nመልስ፡"""
    elif language == "German":
        prompt = f"""Wählen Sie die beste Emotion, die Sie für die gestellte Frage empfinden. Emotionen können nur 'wut', 'freude', 'traurigkeit', 'schuld', 'furcht' oder 'neutral' sein. Geben Sie nur ein einziges Emotionswort aus der Liste der Emotionen ohne weitere Erklärung zurück. \nText: {text}\nAntwort:"""
    elif language == "Hindi":
        prompt = f"""दिए गए प्रश्न के लिए अपनी सबसे अच्छी भावना चुनें। भावनाएँ केवल 'उदासी', 'आनंद', 'अपराध', 'गुस्सा', 'डर', या 'सामान्य' हो सकती हैं। बिना किसी अतिरिक्त स्पष्टीकरण के भावनाओं की सूची से केवल एक ही भावना शब्द लौटाएँ। \nपाठ: {text}\nउत्तर:"""
    # elif country:
    #     prompt = f"You live in {country}, choose the best emotion you feel for the given question. Emotions can be only either 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}"
    else:
        raise ValueError("Either a valid 'language' or 'country' must be provided.")
    
    if prompt_type == "conceptual_chaining":
        prompt = CC_PROMPT + "\n\n" + prompt + "\nAfter </think>, return ONLY the single emotion word without any brackets or extra text."
        
    return prompt


def process_file(tsv_file: str, model: str, get_prediction, language = None, country = None, prompt_type = "standard") -> list[dict]:
    """
    Processes a TSV file and generates predictions for each entry.
    
    Args:
        tsv_file (str): Path to the TSV file
        model (str): Name of the model to use
        get_prediction (callable): Function to get predictions from the model
        language (str, optional): Language for evaluation
        country (str, optional): Country for evaluation
        
    Returns:
        list[dict]: List of results containing predictions and ground truth
    """
    results = []

    with open(tsv_file, "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        next(tsv_reader)  # Skip header row

        for count, row in enumerate(tsv_reader, start=1):
            print(count)

            if language == "English" or country:
                text, gt_emotion, gt_sentiment = row        # only for english tsv
            else:
                text_eng, text, emotion_eng, gt_emotion, sentiment_eng, gt_sentiment = row

            prompt, pred_emotion = get_prediction(model, get_prompt(language, country, text, prompt_type))
            if prompt_type == "conceptual_chaining":
                import re
                cleaned = pred_emotion.lower()
                
                if "</think>" in cleaned:
                    final_ans = cleaned.split("</think>")[-1].strip()
                else:
                    final_ans = cleaned
                    
                match = re.search(r"boxed\[(.*?)\]", final_ans, re.DOTALL)
                if match: final_ans = match.group(1).strip()
                
                valid_emotions = ['anger', 'fear', 'sadness', 'joy', 'guilt', 'neutral']
                found = [e for e in valid_emotions if e in final_ans]
                if found:
                    pred_emotion = found[-1]
                else:
                    pred_emotion = final_ans.strip()
            
            results.append(
                {
                    "prompt": prompt,
                    "text": text,
                    **({"country": country} if country else {"language": language}),
                    "emotion": gt_emotion,
                    "pred_emotion": pred_emotion,
                    "model": model,
                }
            )
    return results


if __name__ == "__main__":
    output_data = process_file(tsv_file=TSV_FILE_PATH, model=MODEL_NAME, get_prediction=get_prediction, language=LANG, country=COUNTRY, prompt_type=PROMPT_TYPE)
    write_json(output_data, OUTPUT_JSON_PATH)


