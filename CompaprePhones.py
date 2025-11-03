# ComparePhones.py — compara inventarios contra text/symbols.py
import json, importlib
symbols = set(importlib.import_module('text.symbols').symbols)

tg = dict(json.load(open('phones_from_textgrid.json', encoding='utf-8')))
lx = dict(json.load(open('phones_from_lexicon.json', encoding='utf-8')))

tg_only = [p for p in tg if p not in symbols]
lx_only = [p for p in lx if p not in symbols]

print("FONEMAS EN TEXTGRID no presentes en symbols:", tg_only[:50], '... total', len(tg_only))
print("FONEMAS EN LEXICÓN no presentes en symbols:", lx_only[:50],  '... total', len(lx_only))
