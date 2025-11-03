import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        # ===== Alineación defensiva de longitudes (evita RuntimeError por desajustes) =====
        B = mel_masks.size(0)

        # 1) Alinear todo lo que es a nivel FONEMA contra src_masks (dur/pitch/energy fonema)
        src_len = src_masks.size(1)

        def _min_l(*tensors):
            return min([t.size(1) for t in tensors])

        # Si tus predicciones/targets existen, recórtalos al mínimo común
        # (algunas ramas pueden no usarse según el feature_level)
        if 'pitch_predictions' in locals() and pitch_predictions is not None:
            L_src_pitch = _min_l(src_masks, pitch_targets, pitch_predictions)
            src_masks      = src_masks[:, :L_src_pitch]
            pitch_targets  = pitch_targets[:, :L_src_pitch]
            pitch_predictions = pitch_predictions[:, :L_src_pitch]

        if 'energy_predictions' in locals() and energy_predictions is not None:
            L_src_energy = _min_l(src_masks, energy_targets, energy_predictions)
            src_masks      = src_masks[:, :L_src_energy]
            energy_targets = energy_targets[:, :L_src_energy]
            energy_predictions = energy_predictions[:, :L_src_energy]

        L_src_dur = _min_l(src_masks, log_duration_targets, log_duration_predictions)
        src_masks             = src_masks[:, :L_src_dur]
        log_duration_targets  = log_duration_targets[:, :L_src_dur]
        log_duration_predictions = log_duration_predictions[:, :L_src_dur]

        # 2) Alinear todo lo que es a nivel FRAME contra mel_masks (mels y, si aplica, pitch/energy frame-level)
        mel_len = mel_masks.size(1)

        def _min_lm(*tensors):
            return min([t.size(1) for t in tensors])

        L_mel = _min_lm(mel_masks, mel_targets, mel_predictions, postnet_mel_predictions)
        mel_masks               = mel_masks[:, :L_mel]
        mel_targets             = mel_targets[:, :L_mel, :]
        mel_predictions         = mel_predictions[:, :L_mel, :]
        postnet_mel_predictions = postnet_mel_predictions[:, :L_mel, :]

        # Si estuvieras usando pitch/energy a nivel FRAME, también recórtalos:
        if self.pitch_feature_level == "frame_level":
            L_mel_pitch = _min_lm(mel_masks, pitch_targets, pitch_predictions)
            mel_masks        = mel_masks[:, :L_mel_pitch]
            mel_targets      = mel_targets[:, :L_mel_pitch, :]  # mantener coherencia con mel
            pitch_targets    = pitch_targets[:, :L_mel_pitch]
            pitch_predictions= pitch_predictions[:, :L_mel_pitch]

        if self.energy_feature_level == "frame_level":
            L_mel_energy = _min_lm(mel_masks, energy_targets, energy_predictions)
            mel_masks        = mel_masks[:, :L_mel_energy]
            mel_targets      = mel_targets[:, :L_mel_energy, :]
            energy_targets   = energy_targets[:, :L_mel_energy]
            energy_predictions = energy_predictions[:, :L_mel_energy]

        # (Opcional) Sanear NaN/Inf antes de pérdidas — versión NO in-place
        def _sanitize(x):
            return None if x is None else torch.nan_to_num(x)

        pitch_targets            = _sanitize(pitch_targets)
        energy_targets           = _sanitize(energy_targets)
        log_duration_targets     = _sanitize(log_duration_targets)
        mel_targets              = _sanitize(mel_targets)

        pitch_predictions        = _sanitize(pitch_predictions)
        energy_predictions       = _sanitize(energy_predictions)
        log_duration_predictions = _sanitize(log_duration_predictions)
        mel_predictions          = _sanitize(mel_predictions)
        postnet_mel_predictions  = _sanitize(postnet_mel_predictions)
        # ===== Fin alineación defensiva =====

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
        )
