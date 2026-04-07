"""
ACT (Action Chunking Transformer) Policy for Robot Manipulation.

ACT learns to predict action sequences (chunks) using a CVAE (Conditional
Variational Autoencoder) with Transformer encoder-decoder architecture.

Key features:
- Action Chunking: Predicts horizon steps of actions at once
- CVAE: Learns action distribution via latent variable z
- Temporal Ensemble: Smooths actions during inference

Reference:
- Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
  (Zhao et al., RSS 2023)
- ACT: Action Chunking with Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *


# ============================================================
# Positional Encoding
# ============================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# Transformer Components
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # Pre-norm + self-attention
        x_norm = self.norm1(x)
        x2, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)
        x = x + x2
        # Pre-norm + feedforward
        x = x + self.ff(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block with cross-attention."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Self-attention with causal mask
        x_norm = self.norm1(x)
        x2, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = x + x2
        # Cross-attention to encoder output
        x_norm = self.norm2(x)
        x2, _ = self.cross_attn(x_norm, memory, memory, attn_mask=memory_mask)
        x = x + x2
        # Feedforward
        x = x + self.ff(self.norm3(x))
        return x


# ============================================================
# CVAE Encoder
# ============================================================

class CVAEEncoder(nn.Module):
    """Encode action sequence to latent z."""

    def __init__(self, action_dim, latent_dim, hidden_dim=256, n_layers=4, nhead=8):
        super().__init__()
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(action_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)

        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, nhead, hidden_dim * 4, dropout=0.1)
            for _ in range(n_layers)
        ])

        # Output to mu and logvar
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, actions):
        """
        Args:
            actions: (B, horizon, action_dim)
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = self.input_proj(actions)  # (B, horizon, hidden_dim)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        # Take the last token's representation
        x = x[:, -1, :]  # (B, hidden_dim)

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


# ============================================================
# CVAE Decoder (Transformer Decoder)
# ============================================================

class CVAEDecoder(nn.Module):
    """Decode latent z + observation to action sequence."""

    def __init__(self, action_dim, latent_dim, cond_dim, hidden_dim=256,
                 n_layers=7, nhead=8, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Learnable action query tokens
        self.action_tokens = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, nhead, hidden_dim * 4, dropout=0.1)
            for _ in range(n_layers)
        ])

        # Output projection to actions
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, z, cond):
        """
        Args:
            z: (B, latent_dim) latent variable
            cond: (B, cond_dim) observation condition
        Returns:
            actions: (B, horizon, action_dim)
        """
        B = z.shape[0]

        # Project latent and condition
        z_embed = self.latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)
        cond_embed = self.cond_proj(cond).unsqueeze(1)  # (B, 1, hidden_dim)

        # Memory: [z, cond]
        memory = torch.cat([z_embed, cond_embed], dim=1)  # (B, 2, hidden_dim)

        # Action queries
        x = self.action_tokens.expand(B, -1, -1)  # (B, horizon, hidden_dim)
        x = self.pos_encoder(x)

        # Causal mask for self-attention
        tgt_mask = torch.triu(
            torch.ones(self.horizon, self.horizon, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)

        # Project to actions
        actions = self.output_proj(x)
        return actions


# ============================================================
# ACT Policy
# ============================================================

class ACTPolicy(BasePolicy):
    """
    ACT (Action Chunking Transformer) Policy.

    Uses CVAE with Transformer encoder-decoder to predict action sequences.
    During training, learns to reconstruct actions conditioned on observations.
    During inference, samples z ~ N(0, I) and decodes action sequence.
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # ACT hyperparameters
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps
        self.latent_dim = policy_cfg.latent_dim
        self.kl_weight = policy_cfg.kl_weight
        self.temporal_agg = policy_cfg.get("temporal_agg", True)
        self.ema_alpha = policy_cfg.get("ema_alpha", 0.5)

        # Verify alignment
        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        # History buffer for inference
        self.obs_history = {}
        self.action_buffer = {}  # For temporal aggregation

        # ---- Observation encoders ----
        self.obs_encoders = nn.ModuleDict()
        obs_dim = 0

        image_embed_size = policy_cfg.image_embed_size
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = image_embed_size
                kwargs.language_dim = policy_cfg.language_encoder.network_kwargs.input_size
                encoder = eval(policy_cfg.image_encoder.network)(**kwargs)
                self.obs_encoders[name] = encoder
                obs_dim += image_embed_size

        self.image_encoders = self.obs_encoders

        text_embed_size = policy_cfg.text_embed_size
        policy_cfg.language_encoder.network_kwargs.output_size = text_embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )
        obs_dim += text_embed_size

        from libero.lifelong.models.bc_rnn_policy import ExtraModalities
        self.extra_encoder = ExtraModalities(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
        )
        obs_dim += self.extra_encoder.extra_low_level_feature_dim

        # LSTM to encode observation sequence
        self.obs_temporal_encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=policy_cfg.obs_hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.final_obs_dim = policy_cfg.obs_hidden_size

        # Action dimension
        self.action_dim = shape_meta["ac_dim"]

        # ---- CVAE Components ----
        self.cvae_encoder = CVAEEncoder(
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_dim=policy_cfg.hidden_dim,
            n_layers=policy_cfg.encoder_n_layers,
            nhead=policy_cfg.nhead,
        )

        self.cvae_decoder = CVAEDecoder(
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            cond_dim=self.final_obs_dim,
            hidden_dim=policy_cfg.hidden_dim,
            n_layers=policy_cfg.decoder_n_layers,
            nhead=policy_cfg.nhead,
            horizon=self.horizon,
        )

        self.to(self.device)

    def encode_obs(self, data):
        """Encode observations to a condition vector (B, obs_hidden_size)."""
        encoded = []

        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape

            if T == 1:
                # Inference: maintain rolling window
                if img_name not in self.obs_history:
                    self.obs_history[img_name] = deque(maxlen=self.n_obs_steps)
                self.obs_history[img_name].append(x.clone())

                history_list = list(self.obs_history[img_name])
                if len(history_list) < self.n_obs_steps:
                    pad_len = self.n_obs_steps - len(history_list)
                    history_list = [history_list[0]] * pad_len + history_list
                x = torch.cat(history_list, dim=1)
                T = x.shape[1]
            else:
                # Training: use first n_obs_steps frames
                x = x[:, :self.n_obs_steps, :, :, :]
                T = self.n_obs_steps

            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"].reshape(B, 1, -1).repeat(1, T, 1).reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

        # Extra modalities
        extra = self.extra_encoder(data["obs"])
        T_extra = extra.shape[1]

        if T_extra == 1:
            if "extra" not in self.obs_history:
                self.obs_history["extra"] = deque(maxlen=self.n_obs_steps)
            self.obs_history["extra"].append(extra.clone())

            history_list = list(self.obs_history["extra"])
            if len(history_list) < self.n_obs_steps:
                pad_len = self.n_obs_steps - len(history_list)
                history_list = [history_list[0]] * pad_len + history_list
            extra = torch.cat(history_list, dim=1)
        else:
            extra = extra[:, :self.n_obs_steps, :]

        encoded.append(extra)

        lang_h = self.language_encoder(data)
        T_obs = encoded[0].shape[1]
        encoded.append(lang_h.unsqueeze(1).expand(-1, T_obs, -1))

        x = torch.cat(encoded, dim=-1)
        _, (h_n, _) = self.obs_temporal_encoder(x)
        return h_n[-1]

    def forward(self, data, train_mode=True):
        """
        Training forward: CVAE reconstruction loss.

        For training:
        1. Encode actions to z ~ q(z|a)
        2. Decode z with observation condition to reconstruct actions
        3. Loss = reconstruction loss + KL divergence
        """
        data = self.preprocess_input(data, train_mode=train_mode)
        obs_cond = self.encode_obs(data)

        # Action target
        start_idx = self.n_obs_steps - 1
        action = data["actions"][:, start_idx : start_idx + self.horizon, :]
        B = action.shape[0]

        # Encode actions to latent z
        mu, logvar = self.cvae_encoder(action)
        z = self.cvae_encoder.reparameterize(mu, logvar)

        # Decode z with observation condition
        action_pred = self.cvae_decoder(z, obs_cond)

        return {
            "action": action,
            "action_pred": action_pred,
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(self, data, reduction="mean"):
        output = self.forward(data, train_mode=True)

        # Reconstruction loss (L1 or MSE)
        recon_loss = F.l1_loss(output["action_pred"], output["action"], reduction=reduction)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + output["logvar"] - output["mu"].pow(2) -
                                    output["logvar"].exp(), dim=-1)
        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss

    def get_action(self, data):
        """
        Inference: sample z ~ N(0, I) and decode.
        Returns: (B, action_dim) numpy array
        """
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            obs_cond = self.encode_obs(data)
            B = obs_cond.shape[0]

            # Deterministic z (mean of prior N(0, I))
            z = torch.zeros(B, self.latent_dim, device=obs_cond.device)

            # Decode action sequence
            actions = self.cvae_decoder(z, obs_cond)  # (B, horizon, action_dim)

            # Temporal aggregation (optional)
            if self.temporal_agg:
                if "step_idx" not in self.action_buffer:
                    # Start new chunk
                    self.action_buffer["actions"] = list(actions[0])  # List of horizon actions
                    self.action_buffer["step_idx"] = 0

                step_idx = self.action_buffer["step_idx"]
                buf = self.action_buffer["actions"]

                if step_idx >= len(buf):
                    # Chunk exhausted, start new one
                    self.action_buffer["actions"] = list(actions[0])
                    self.action_buffer["step_idx"] = 0
                    step_idx = 0

                action = self.action_buffer["actions"][step_idx].cpu().numpy()
                self.action_buffer["step_idx"] = step_idx + 1
            else:
                if "actions" not in self.action_buffer or len(self.action_buffer["actions"]) == 0:
                    self.action_buffer["actions"] = list(actions[0])

                action = self.action_buffer["actions"].pop(0).cpu().numpy()

        return action.reshape(1, -1)  # (1, action_dim)

    def reset(self):
        """Clear observation history and action buffer."""
        self.obs_history = {}
        self.action_buffer = {}
