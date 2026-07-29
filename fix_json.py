import os
import json
import csv
import glob
import sys
sys.path.append('.')
from inference_utils import _parse_row

LANGUAGE_MAP = {
    'eng': 'English',
    'ara': 'Arabic',
    'spn': 'Spanish',
    'deu': 'German',
    'amh': 'Amharic',
    'hin': 'Hindi'
}

for jpath in glob.glob('outputs/*/lang/*.json'):
    filename = os.path.basename(jpath)
    prefix = filename.split('_')[0]
    if prefix not in LANGUAGE_MAP: continue
    
    lang = LANGUAGE_MAP[prefix]
    if lang == 'English': continue
    
    tsv_path = f'data/test/{prefix}.tsv'
    if not os.path.exists(tsv_path): continue
    
    correct_emotions = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        for row in reader:
            if not row or len(row) < 3: continue
            _, correct_emotion = _parse_row(row, lang, None)
            correct_emotions.append(correct_emotion)
            
    with open(jpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if len(data) != len(correct_emotions):
        print(f'Length mismatch in {jpath}: json {len(data)} != tsv {len(correct_emotions)}')
        continue
        
    changed = 0
    for i, item in enumerate(data):
        if item.get('emotion') != correct_emotions[i]:
            item['emotion'] = correct_emotions[i]
            changed += 1
            
    if changed > 0:
        with open(jpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Fixed {changed} items in {jpath}')
