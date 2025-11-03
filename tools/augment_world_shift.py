import os, numpy as np, soundfile as sf, pyworld as pw, argparse, shutil
from pathlib import Path

def shift_f0_keep_formants(y, sr, semitones=1.0):
    _f0, t = pw.dio(y.astype(np.float64), sr, frame_period=1000*256/sr)
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
    sp = pw.cheaptrick(y.astype(np.float64), f0, t, sr)
    ap = pw.d4c(y.astype(np.float64), f0, t, sr)
    f0_shift = f0 * (2.0**(semitones/12.0))
    y_hat = pw.synthesize(f0_shift, sp, ap, sr).astype(np.float32)
    # normaliza leve
    mx = max(1e-6, np.max(np.abs(y_hat))); y_hat = y_hat / mx * 0.95
    return y_hat

def main(raw_src, raw_dst, speaker_src="angelina", speaker_dst="angelina_f1", list_txt=None, semitones=1.0):
    raw_src = Path(raw_src); raw_dst = Path(raw_dst)
    sdir = raw_src/speaker_src; ddir = raw_dst/speaker_dst
    ddir.mkdir(parents=True, exist_ok=True)
    # Si pasas una lista (train_balanced.txt), sólo procesa esas bases
    allow = None
    if list_txt and Path(list_txt).exists():
        allow = set([ln.split("|",1)[0] for ln in open(list_txt,encoding='utf-8') if ln.strip()])

    for fn in os.listdir(sdir):
        if not fn.lower().endswith(".wav"): continue
        base = fn[:-4]
        if allow and base not in allow: continue
        y, sr = sf.read(str(sdir/fn))
        if y.ndim>1: y = y.mean(1)
        y2 = shift_f0_keep_formants(y, sr, semitones=semitones)
        sf.write(str(ddir/fn), y2, sr, subtype='PCM_16')
        # copia .lab asociado si existe
        lab = sdir/(base+".lab")
        if lab.exists(): shutil.copy2(lab, ddir/(base+".lab"))
    print("Aumentados en:", ddir)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw_src", required=True)
    ap.add_argument("--raw_dst", required=True)
    ap.add_argument("--speaker_src", default="angelina")
    ap.add_argument("--speaker_dst", default="angelina_f1")
    ap.add_argument("--list_txt", default=None)
    ap.add_argument("--semitones", type=float, default=1.0)
    a=ap.parse_args()
    main(a.raw_src, a.raw_dst, a.speaker_src, a.speaker_dst, a.list_txt, a.semitones)
