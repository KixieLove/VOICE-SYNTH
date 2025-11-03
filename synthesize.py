import re
import argparse
from string import punctuation
import io


import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from phonemizer import phonemize
from text.symbols import symbols as INVENTORY

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence
from text.symbols import symbols as MODEL_SYMBOLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
def read_lexicon(path):
    lexicon = {}
    # intenta encodings en orden
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc, errors="strict") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    word = parts[0].lower()
                    phones = parts[1:]
                    lexicon[word] = phones
            break  # leído bien → salimos del bucle
        except UnicodeDecodeError:
            continue
    return lexicon
"""

_SP_RE = re.compile(r"([,;.\-\?\!\s+])")

from text.symbols import symbols as INVENTORY

def map_to_inventory(phones):
    mapped = []
    for p in phones:
        if p == "sp" and "@sp" in INVENTORY:
            mapped.append("sp")
        elif p == "spn" and "@spn" in INVENTORY:
            mapped.append("spn")
        elif p in INVENTORY:
            mapped.append(p)
        else:
            mapped.append(p)  # deja pasar para ver warnings si algo no cuadra
    return mapped


# === Pausas audibles sin reentrenar ===
# IMPORTANTE: usar 'sp' SIN @; text_to_sequence añadirá '@' dentro de {…}
PAUSE_TOKEN = "spn"

def stretch_pauses(phones, repeat=3):
    out = []
    for p in phones:
        if p == PAUSE_TOKEN:
            out.extend([p] * repeat)  # repite 'sp' para alargar pausa
        else:
            out.append(p)
    return out
# === fin pausas ===



def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0].lower()
                phones = parts[1:]
                lexicon[word] = phones
    return lexicon

def _phonemize_es(word: str) -> list:
    # Phonemizer produce AFI con espacios; normalizamos algunas variantes
    ph = phonemize(word, language="es", backend="espeak", strip=True)
    ph = ph.replace("  ", " ").strip()

    # Normalizaciones mínimas para que coincida con tu set:
    # - 'g' ASCII -> 'ɡ' (U+0261)
    ph = ph.replace("g", "ɡ")
    # - espeak a veces separa ɟ y ʝ; únelos si aparecen consecutivos
    ph = ph.replace("ɟ ʝ", "ɟʝ")

    # Puedes añadir aquí otras reglas si detectas discrepancias puntuales
    return ph.split()

def preprocess_spanish(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    for w in _SP_RE.split(text):
        if not w or w.isspace():
            continue
        lw = w.lower()
        if lw in lexicon:
            phones += lexicon[lw]
        elif _SP_RE.fullmatch(w):         # signos/espacios -> silencio corto
            phones += ["spn"]
        else:
            phones += _phonemize_es(lw)

    # Mapear a inventario personalizado
    phones = map_to_inventory(phones)

    # Fuerza pausas audibles en inferencia (no cambia tu dataset/ckpt)
    phones = stretch_pauses(phones, repeat=1)


    # Formato {a b c} que ya espera tu pipeline
    ph_seq = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence:", text)
    print("Phoneme Sequence:", ph_seq)

    # **Comprobación**: avisa si algún token no existe en MODEL_SYMBOLS
    unknown = sorted({p for p in phones if p not in MODEL_SYMBOLS})
    if unknown:
        print("WARN: símbolos fuera de tu inventario:", unknown[:30])

    seq = np.array(
        text_to_sequence(
            ph_seq, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return seq

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)

def safe_filename(name, maxlen=80):
    # quita caracteres inválidos en Windows y espacios/puntos finales
    name = name.replace('\n', ' ').strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    name = name.rstrip('. ')
    return (name[:maxlen] or "sample")


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            # ==== DEBUG DURACIONES (temporal) ====
            dur = output[5].long().cpu().numpy()  # duration_rounded
            ids_np = batch[3].cpu().numpy()[0][: batch[4].cpu().numpy()[0]]

            from text.symbols import symbols as SYM
            syms = [SYM[i] for i in ids_np]
            print("== DUR POR FONEMA ==")
            for s, d in zip(syms, dur[0][:len(syms)]):
                print(f"{s:>6} -> {int(d)}")
            # ==== FIN DEBUG ====

            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = [safe_filename(args.text[:100])]   # <- se usa para el nombre de archivo
        raw_texts = [args.text[:100]]            # <- conserva el texto original para logs

        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_spanish(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
