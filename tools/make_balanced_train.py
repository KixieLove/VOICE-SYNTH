import os, json, numpy as np, csv, argparse, shutil
from pathlib import Path
def load_stats(preproc):
    with open(Path(preproc,'stats.json'),'r',encoding='utf-8') as f: return json.load(f)

def main(preproc_dir, keep_ratio=1.0, soft_ratio=0.35, out_name="train_balanced.txt"):
    pre = Path(preproc_dir)
    train_txt = pre/'train.txt'
    rows = [ln.strip() for ln in open(train_txt, encoding='utf-8') if ln.strip()]
    items = []
    for ln in rows:
        base, spk, ph, raw = ln.split('|', 3)
        p = np.load(pre/'pitch'/f'{spk}-pitch-{base}.npy')
        e = np.load(pre/'energy'/f'{spk}-energy-{base}.npy')
        nz = p[p>0]
        f0_med = float(np.median(nz)) if nz.size>0 else 0.0
        e_mean = float(np.mean(e)) if e.size>0 else 0.0
        items.append((base,spk,ph,raw,f0_med,e_mean))

    # z-scores para mezclar criterios
    f0s = np.array([x[4] for x in items]); ems = np.array([x[5] for x in items])
    f0z = (f0s - f0s.mean())/(f0s.std()+1e-8)
    ez  = (ems - ems.mean())/(ems.std()+1e-8)

    # score alto = F0 alto y energía baja (0.7 es peso de energía, puedes ajustar)
    score = f0z - 0.7*ez
    order = np.argsort(-score)  # descendente

    n = int(len(items)*keep_ratio)
    top = order[:int(n*soft_ratio)]
    rest = order[int(n*soft_ratio):n]

    # Escribe CSV diagnóstico
    with open(pre/'balance_diag.csv','w',newline='',encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['base','spk','f0_med','energy_mean','score'])
        for i in order: w.writerow([items[i][0],items[i][1],items[i][4],items[i][5],float(score[i])])

    # Construye lista balanceada: duplica los “soft”
    out = []
    for i in top:  out.append("|".join(items[i][:4]))
    for i in top:  out.append("|".join(items[i][:4]))         # duplicado = más probabilidad
    for i in rest: out.append("|".join(items[i][:4]))

    # Backup y reemplazo “train.txt” por el balanceado (o escribe un nombre alterno)
    with open(pre/out_name,'w',encoding='utf-8') as f: f.write("\n".join(out))
    print("Escrito:", pre/out_name, " filas:", len(out))
    # Si quieres que train.py lo use sin tocar código, reemplaza train.txt:
    shutil.copy2(pre/out_name, pre/'train.txt'); print("Reemplazado train.txt")
    if len(out) == 0:
        print("ABORT: no se encontraron filas para entrenar (train vacío). No toco train.txt.")
        return
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--keep_ratio", type=float, default=1.0)
    ap.add_argument("--soft_ratio", type=float, default=0.35)
    ap.add_argument("--out_name", default="train_balanced.txt")
    a=ap.parse_args()
    main(a.preproc, a.keep_ratio, a.soft_ratio, a.out_name)

