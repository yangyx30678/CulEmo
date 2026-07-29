import os
import json
import glob
import re

TRANSLATION_MAP = {
    # Arabic
    "غضب": "anger", "خوف": "fear", "حزن": "sadness", "فرح": "joy", "ذنب": "guilt", "محايد": "neutral",
    # Spanish
    "enojo": "anger", "miedo": "fear", "tristeza": "sadness", "alegría": "joy", "alegria": "joy", "culpa": "guilt", "neutral": "neutral",
    # Amharic
    "ቁጣ": "anger", "ፍርሀት": "fear", "ሀዘን": "sadness", "ደስታ": "joy", "ጥፋተኛ": "guilt", "መደበኛ": "neutral",
    # German
    "wut": "anger", "furcht": "fear", "traurigkeit": "sadness", "freude": "joy", "schuld": "guilt", 
    # Hindi
    "गुस्सा": "anger", "डर": "fear", "उदासी": "sadness", "आनंद": "joy", "अपराध": "guilt", "सामान्य": "neutral"
}

def clean_pred(pred):
    if not pred: return "error"
    # lowercase and strip punctuation
    p = pred.lower().strip(' .,"\'\n\t![]{}()')
    
    # Try exact match first
    if p in TRANSLATION_MAP:
        return TRANSLATION_MAP[p]
    
    # Try substring match
    for native, eng in TRANSLATION_MAP.items():
        if native in p:
            return eng
            
    # If already English, just return it
    if p in ["anger", "fear", "sadness", "joy", "guilt", "neutral"]:
        return p
        
    return p # return as is if no match found

for jpath in glob.glob('outputs/*/lang/*.json'):
    with open(jpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    changed = 0
    for item in data:
        old_pred = item.get('pred_emotion', '')
        new_pred = clean_pred(old_pred)
        if old_pred != new_pred:
            item['pred_emotion'] = new_pred
            changed += 1
            
    if changed > 0:
        with open(jpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Translated {changed} predictions in {jpath}')
