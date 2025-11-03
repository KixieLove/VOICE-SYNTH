# LexPhones.py — fonemas presentes en tu lexicón
import re, json
lex = 'lexicon/new_spanish.txt'  # ajusta si usas otro archivo
phones = {}
bad = 0
with open(lex, encoding='utf-8') as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        parts = re.split(r'\s+', ln)
        if len(parts) < 2:
            bad += 1
            continue
        for ph in parts[1:]:
            phones[ph] = phones.get(ph, 0) + 1

with open('phones_from_lexicon.json','w',encoding='utf-8') as f:
    json.dump(sorted(phones.items(), key=lambda x: -x[1]), f, ensure_ascii=False, indent=2)

print('Escribí phones_from_lexicon.json; líneas inválidas en lexicón:', bad)
