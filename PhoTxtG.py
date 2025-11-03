# PhoTxtG.py  —  extrae los fonemas reales desde los TextGrid de MFA
import os, json, yaml, tgt
from collections import Counter

cfg = yaml.safe_load(open('config/LJSpeech/preprocess.yaml', encoding='utf-8'))
tg_root = os.path.join(cfg['path']['preprocessed_path'], 'TextGrid')

phone_counts = Counter()
n_files = 0
missing_tier = []

for spk in os.listdir(tg_root):
    spk_dir = os.path.join(tg_root, spk)
    if not os.path.isdir(spk_dir):
        continue
    for fn in os.listdir(spk_dir):
        if not fn.endswith('.TextGrid'):
            continue
        n_files += 1
        tg = tgt.io.read_textgrid(os.path.join(spk_dir, fn))
        try:
            tier = tg.get_tier_by_name('phones')
        except KeyError:
            missing_tier.append(os.path.join(spk_dir, fn))
            continue

        # tier._objects es la lista de Interval
        for interval in tier._objects:
            p = (interval.text or "").strip()
            if p:
                phone_counts[p] += 1

out_path = 'phones_from_textgrid.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(phone_counts.most_common(), f, ensure_ascii=False, indent=2)

print(f"Procesados TextGrid: {n_files}")
print(f"Fonemas distintos: {len(phone_counts)}")
print(f"Archivo generado: {out_path}")
if missing_tier:
    print(f'AVISO: {len(missing_tier)} archivos sin tier "phones" (lista en missing_phones_tier.json)')
    with open('missing_phones_tier.json', 'w', encoding='utf-8') as f:
        json.dump(missing_tier, f, ensure_ascii=False, indent=2)
