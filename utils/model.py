import os
import json

import torch
import numpy as np
import librosa

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)

    ckpt = None
    if getattr(args, "restore_step", None):
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            f"{args.restore_step}.pth.tar",
        )
        # map_location=device evita errores si el ckpt fue guardado en otra GPU/CPU
        ckpt = torch.load(ckpt_path, map_location=device)

        # --- parche: redimensionar embeddings si cambió el vocabulario ---
        sd = ckpt["model"]
        cur = model.state_dict()

        def _resize_and_copy(key):
            if key in sd and key in cur and sd[key].shape != cur[key].shape:
                new = cur[key].clone()    # tensor con el tamaño nuevo
                old = sd[key]             # tensor del checkpoint
                rows = min(new.shape[0], old.shape[0])
                new[:rows] = old[:rows]   # copia lo que coincide
                sd[key] = new             # filas nuevas quedan inicializadas
                return True
            return False

        # embedding de entrada de fonemas (clave típica en FastSpeech2)
        _resize_and_copy("encoder.src_word_emb.weight")

        # (opcional) si tu modelo es multi-speaker y cambió el # de speakers:
        for k in ["speaker_emb.weight", "speaker_embedding.weight"]:
            _resize_and_copy(k)

        # permite claves que no coincidan exactamente de tamaño/forma
        model.load_state_dict(sd, strict=False)
        # --- fin parche ---

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, getattr(args, "restore_step", 0)
        )
        #if ckpt is not None and "optimizer" in ckpt:
         #   scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_(False)
    return model



def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if config["vocoder"]["model"] == "Griffin":
        return None


    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)

    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            hcfg = json.load(f)
        hcfg = hifigan.AttrDict(hcfg)
        vocoder = hifigan.Generator(hcfg)

        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar", map_location=device)
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar", map_location=device)

        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "Griffin":
            # mels: Tensor [B, 80, T] en log natural; convertir a numpy
            if isinstance(mels, torch.Tensor):
                m_np = mels.detach().cpu().numpy()
            else:
                m_np = mels

            sr   = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
            nfft = preprocess_config["preprocessing"]["stft"]["filter_length"]
            hop  = preprocess_config["preprocessing"]["stft"]["hop_length"]
            win  = preprocess_config["preprocessing"]["stft"]["win_length"]
            fmin = preprocess_config["preprocessing"]["mel"]["mel_fmin"]
            fmax = preprocess_config["preprocessing"]["mel"]["mel_fmax"]

            out = []
            for mel in m_np:  # [80, T]
                # inversa del log (natural) -> magnitud mel lineal
                mel_lin = np.exp(mel)
                # mel -> STFT y luego Griffin-Lim
                S = librosa.feature.inverse.mel_to_stft(
                    mel_lin, sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, power=1.0
                )
                y = librosa.griffinlim(
                    S, n_iter=60, hop_length=hop, win_length=win, n_fft=nfft
                )
                y = (y * preprocess_config["preprocessing"]["audio"]["max_wav_value"]).astype("int16")
                out.append(y)
            return out

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    # if lengths is not None:  # (opcional) recortar a longitudes originales
    #     for i in range(len(wavs)):
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs
