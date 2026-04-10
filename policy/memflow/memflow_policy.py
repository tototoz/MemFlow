"""
Memory-Augmented Flow Matching Policy (MemFlow).

Extends FlowMatchingPolicy with hierarchical memory:
- L1 Working Memory: Causal Transformer over recent (obs, action) pairs for short-term motion coherence
- L2 Episodic Memory: Event-driven sparse memory for mid-term task progress tracking

Key design: Training and inference use the SAME memory update logic.

Reference:
- Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
- MemFlow: Memory-Augmented Flow Policy for Long-Horizon Manipulation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops.layers.torch import Rearrange
from collections import deque

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *


# ============================================================
# DINOv2 Feature Extractor for Event Detection
# ============================================================

class DINOv2FeatureExtractor(nn.Module):
    """
    Frozen DINOv2 feature extractor for event detection.

    Uses pre-trained DINOv2 to extract semantic features from images,
    which are then used to detect "key events" based on feature distance.
    """
    def __init__(self, model_name="dinov2_small", freeze=True, local_path=None):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        try:
            from transformers import AutoModel

            if local_path is not None and os.path.exists(local_path):
                print(f"[DINOv2] 从本地加载: {local_path}")
                self.model = AutoModel.from_pretrained(local_path)
            elif os.path.exists(model_name):
                print(f"[DINOv2] 从本地加载: {model_name}")
                self.model = AutoModel.from_pretrained(model_name)
            else:
                # DINOv2 在 Hugging Face 上的名称用连字符
                hf_model_name = model_name.replace("_", "-")
                print(f"[DINOv2] 从 Hugging Face 下载: facebook/{hf_model_name}")
                self.model = AutoModel.from_pretrained(f"facebook/{hf_model_name}")

            self.feature_dim = self.model.config.hidden_size
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers\n"
                "DINOv2 will be downloaded automatically on first use."
            )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # Distance history for adaptive threshold (used in inference)
        self.distance_history = deque(maxlen=100)
        self.event_threshold = None

    def extract_features(self, image):
        """
        Extract DINOv2 features from image.

        Args:
            image: (B, C, H, W) tensor, RGB format, values in [0, 1]

        Returns:
            (B, feature_dim) feature vectors
        """
        _, _, H, W = image.shape
        if H != 224 or W != 224:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        image = (image - mean) / std

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(pixel_values=image)
                features = outputs.last_hidden_state[:, 0, :]
        else:
            outputs = self.model(pixel_values=image)
            features = outputs.last_hidden_state[:, 0, :]

        return features

    def extract_features_batch(self, images):
        """
        Extract DINOv2 features from a batch of images.

        Args:
            images: (B, T, C, H, W) tensor

        Returns:
            (B, T, feature_dim) feature vectors
        """
        B, T, C, H, W = images.shape
        images = images.reshape(B * T, C, H, W)
        features = self.extract_features(images)
        return features.reshape(B, T, -1)

    def detect_events_from_sequence(self, features, threshold_percentile=95):
        """
        Detect events from a sequence of features (for training).

        Args:
            features: (B, T, feature_dim) feature sequence
            threshold_percentile: percentile for event detection

        Returns:
            event_indices: list of indices where events occur
            distances: (T-1,) distances between consecutive frames
        """
        _, T, _ = features.shape

        # Handle single-frame case
        if T == 1:
            return [0], np.array([])

        # Compute distances between consecutive frames
        distances = torch.norm(features[:, 1:, :] - features[:, :-1, :], dim=-1)  # (B, T-1)
        distances = distances.mean(dim=0).cpu().numpy()  # (T-1,)

        # Compute threshold
        threshold = np.percentile(distances, threshold_percentile)

        # Find event indices (where distance > threshold)
        event_indices = [0]  # First frame is always an event
        for i, d in enumerate(distances):
            if d > threshold:
                event_indices.append(i + 1)

        return event_indices, distances

    def detect_event_single(self, feature, prev_feature, threshold_percentile=95):
        """
        Detect if current frame is a "key event" (for inference).

        Args:
            feature: (B, D) current frame feature
            prev_feature: (B, D) previous frame feature, or None
            threshold_percentile: percentile for event detection

        Returns:
            is_event: bool
            distance: float
        """
        if prev_feature is None:
            return True, 0.0

        distance = torch.norm(feature - prev_feature, dim=-1).mean().item()
        self.distance_history.append(distance)

        if len(self.distance_history) >= 10:
            distances = np.array(self.distance_history)
            self.event_threshold = np.percentile(distances, threshold_percentile)

        if self.event_threshold is not None:
            is_event = distance > self.event_threshold
        else:
            is_event = distance > 0.1

        return is_event, distance


# ============================================================
# UNet building blocks
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim=None, time_embed_dim=256,
                 down_dims=(128, 256, 512), kernel_size=3, n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        cond_dim = time_embed_dim + (global_cond_dim or 0)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.down_modules = down_modules
        self.up_modules = up_modules

    def forward(self, sample, t, global_cond=None):
        x = einops.rearrange(sample, 'b h t -> b t h')

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=x.device)
        elif t.ndim == 0:
            t = t[None].to(x.device)
        t = t.expand(x.shape[0])

        global_feature = self.time_encoder(t)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


# ============================================================
# L1 Working Memory: Causal Transformer
# ============================================================

class WorkingMemory(nn.Module):
    """
    L1 Working Memory for short-term motion coherence.

    Uses a Causal Transformer to encode recent (obs_feature, action) pairs.
    """
    def __init__(self, obs_dim, action_dim, memory_len=16, hidden_dim=256,
                 n_heads=4, n_layers=2, output_dim=256):
        super().__init__()
        self.memory_len = memory_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.obs_embed = nn.Linear(obs_dim, hidden_dim)
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        self.pos_embed = nn.Embedding(memory_len * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.register_buffer('causal_mask', self._generate_causal_mask())

    def _generate_causal_mask(self):
        seq_len = self.memory_len * 2
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, obs_features, actions):
        """
        Args:
            obs_features: (B, T, obs_dim)
            actions: (B, T, action_dim)

        Returns:
            (B, output_dim)
        """
        B, T, _ = obs_features.shape

        if T > self.memory_len:
            obs_features = obs_features[:, -self.memory_len:, :]
            actions = actions[:, -self.memory_len:, :]
            T = self.memory_len

        obs_tokens = self.obs_embed(obs_features)
        act_tokens = self.action_embed(actions)

        tokens = torch.stack([obs_tokens, act_tokens], dim=2)
        tokens = tokens.reshape(B, T * 2, self.hidden_dim)

        positions = torch.arange(T * 2, device=tokens.device)
        tokens = tokens + self.pos_embed(positions)

        # Causal mask: True means "don't attend"
        mask = self.causal_mask[:T*2, :T*2]
        encoded = self.transformer(tokens, mask=mask)

        memory = self.output_proj(encoded[:, -1, :])
        return memory


# ============================================================
# L2 Episodic Memory: Event-driven Sparse Memory
# ============================================================

class EpisodicMemory(nn.Module):
    """
    L2 Episodic Memory for mid-term task progress tracking.

    Uses cross-attention to retrieve relevant memories from event bank.
    """
    def __init__(self, dino_feature_dim=384, hidden_dim=256, output_dim=256,
                 n_heads=4, max_events=32, event_threshold_percentile=95):
        super().__init__()
        self.dino_feature_dim = dino_feature_dim
        self.hidden_dim = hidden_dim
        self.max_events = max_events
        self.event_threshold_percentile = event_threshold_percentile

        self.key_proj = nn.Linear(dino_feature_dim, hidden_dim)
        self.value_proj = nn.Linear(dino_feature_dim, hidden_dim)
        self.query_proj = nn.Linear(dino_feature_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, dino_feature, memory_bank):
        """
        Args:
            dino_feature: (B, D) current DINOv2 feature
            memory_bank: (B, N, D) stored event features

        Returns:
            (B, output_dim)
        """
        B = dino_feature.shape[0]

        if memory_bank is None or memory_bank.shape[1] == 0:
            return torch.zeros(B, self.output_proj.out_features, device=dino_feature.device)

        query = self.query_proj(dino_feature).unsqueeze(1)
        keys = self.key_proj(memory_bank)
        values = self.value_proj(memory_bank)

        attended, _ = self.cross_attn(query, keys, values)
        memory = self.output_proj(attended.squeeze(1))

        return memory


# ============================================================
# MemFlow Policy
# ============================================================

class MemFlowPolicy(BasePolicy):
    """
    Memory-Augmented Flow Matching Policy.

    Key design: Training and inference use the SAME memory update logic.

    - L1 Working Memory: Causal Transformer over recent (obs, action) pairs
    - L2 Episodic Memory: DINOv2-based event detection + Cross-Attention
    """
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # Flow Matching parameters
        self.num_integration_steps = policy_cfg.get("num_integration_steps", 10)
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps

        # Memory toggles
        self.use_working_memory = policy_cfg.get("use_working_memory", False)
        self.use_episodic_memory = policy_cfg.get("use_episodic_memory", False)
        self.use_dinov2_events = policy_cfg.get("use_dinov2_events", True)

        # Verify alignment
        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        # Inference buffers
        self.obs_history = {}
        self.obs_feature_history = deque(maxlen=16)  # Store obs_feature for working memory
        self.action_history = deque(maxlen=16)
        self.episodic_memory_bank = None
        self.prev_dino_feature = None  # For event detection in inference
        self.dino_feature_history = deque(maxlen=100)  # Store DINOv2 features for event history

        # ---- DINOv2 Feature Extractor ----
        if self.use_episodic_memory and self.use_dinov2_events:
            dinov2_model = policy_cfg.get("dinov2_model", "dinov2_small")
            dinov2_local_path = policy_cfg.get("dinov2_local_path", None)
            self.dino_extractor = DINOv2FeatureExtractor(
                model_name=dinov2_model,
                freeze=True,
                local_path=dinov2_local_path,
            )
            self.dino_feature_dim = self.dino_extractor.feature_dim
            print(f"[MemFlow] DINOv2 loaded, feature_dim={self.dino_feature_dim}")
        else:
            self.dino_extractor = None
            self.dino_feature_dim = 384

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

        self.obs_temporal_encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=policy_cfg.obs_hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.final_obs_dim = policy_cfg.obs_hidden_size
        self.obs_dim = obs_dim  # Store for later use

        self.action_dim = shape_meta["ac_dim"]

        # ---- Memory Modules ----
        memory_dim = policy_cfg.get("memory_dim", 256)
        self.memory_dim = memory_dim

        if self.use_working_memory:
            self.working_memory = WorkingMemory(
                obs_dim=self.final_obs_dim,
                action_dim=self.action_dim,
                memory_len=policy_cfg.get("working_memory_len", 16),
                hidden_dim=policy_cfg.get("working_memory_hidden_dim", 256),
                n_heads=policy_cfg.get("working_memory_n_heads", 4),
                n_layers=policy_cfg.get("working_memory_n_layers", 2),
                output_dim=memory_dim,
            )
            print("[MemFlow] L1 Working Memory enabled")

        if self.use_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                dino_feature_dim=self.dino_feature_dim,
                hidden_dim=policy_cfg.get("episodic_memory_hidden_dim", 256),
                output_dim=memory_dim,
                n_heads=policy_cfg.get("episodic_memory_n_heads", 4),
                max_events=policy_cfg.get("max_episodic_events", 32),
                event_threshold_percentile=policy_cfg.get("event_threshold_percentile", 95),
            )
            print("[MemFlow] L2 Episodic Memory enabled")

        # ---- Vector Field Network ----
        cond_dim = self.final_obs_dim
        if self.use_working_memory:
            cond_dim += memory_dim
        if self.use_episodic_memory:
            cond_dim += memory_dim

        self.vector_field_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=cond_dim,
            time_embed_dim=policy_cfg.get("time_embed_dim", 256),
            down_dims=policy_cfg.down_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
            cond_predict_scale=policy_cfg.cond_predict_scale,
        )

        self.to(self.device)

        if self.dino_extractor is not None:
            self.dino_extractor.to(self.device)

    def encode_obs_sequence(self, data):
        """
        Encode observation sequence to feature sequence.

        This is used for BOTH training and inference to ensure consistency.

        Returns:
            obs_features: (B, T, final_obs_dim) sequence of observation features
        """
        encoded = []

        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape

            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"].reshape(B, 1, -1).repeat(1, T, 1).reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

        T_obs = encoded[0].shape[1]

        # Extra modalities - handle inference history for dimension alignment
        extra = self.extra_encoder(data["obs"])
        if extra.shape[1] == 1 and T_obs > 1:
            # Inference: expand to match image encoder history length
            extra = extra.expand(-1, T_obs, -1)
        encoded.append(extra)

        lang_h = self.language_encoder(data)
        encoded.append(lang_h.unsqueeze(1).expand(-1, T_obs, -1))

        x = torch.cat(encoded, dim=-1)  # (B, T, obs_dim)

        # LSTM over sequence
        output, (h_n, _) = self.obs_temporal_encoder(x)  # output: (B, T, hidden_size)

        return output  # (B, T, final_obs_dim)

    def extract_dino_features_from_data(self, data):
        """
        Extract DINOv2 features from observation data.

        Returns:
            dino_features: (B, T, dino_feature_dim) or None
        """
        if self.dino_extractor is None:
            return None

        for img_name in self.image_encoders.keys():
            if "rgb" in img_name:
                x = data["obs"][img_name]  # (B, T, C, H, W)
                # DINOv2 expects values in [0, 1], assume input is already normalized
                dino_features = self.dino_extractor.extract_features_batch(x)
                return dino_features

        return None

    def build_episodic_memory_bank(self, dino_features, event_threshold_percentile=95):
        """
        Build episodic memory bank from DINOv2 feature sequence.

        Used in BOTH training and inference for consistency.

        Args:
            dino_features: (B, T, D) DINOv2 features
            event_threshold_percentile: percentile for event detection

        Returns:
            memory_bank: (B, N, D) where N is number of events
        """
        _, T, _ = dino_features.shape

        # Detect events
        event_indices, _ = self.dino_extractor.detect_events_from_sequence(
            dino_features, event_threshold_percentile
        )

        # Build memory bank from event frames
        if len(event_indices) == 0:
            event_indices = [0]  # At least use first frame

        # Limit to max_events
        if len(event_indices) > self.episodic_memory.max_events:
            # Keep most recent events
            event_indices = event_indices[-self.episodic_memory.max_events:]

        # Gather event features
        event_features = []
        for idx in event_indices:
            if idx < T:
                event_features.append(dino_features[:, idx:idx+1, :])

        if len(event_features) > 0:
            memory_bank = torch.cat(event_features, dim=1)  # (B, N, D)
        else:
            memory_bank = dino_features[:, 0:1, :]  # Use first frame

        return memory_bank

    def compute_memory_condition(self, obs_feature_seq, action_seq, dino_feature_seq):
        """
        Compute memory-augmented condition vector.

        Unified function for BOTH training and inference.

        Args:
            obs_feature_seq: (B, T, final_obs_dim) observation feature sequence
            action_seq: (B, T, action_dim) action sequence
            dino_feature_seq: (B, T, dino_dim) DINOv2 feature sequence

        Returns:
            cond: (B, cond_dim) condition vector
        """
        B = obs_feature_seq.shape[0]

        # Get current obs feature (last in sequence)
        current_obs_feature = obs_feature_seq[:, -1, :]  # (B, final_obs_dim)
        cond = [current_obs_feature]

        # L1 Working Memory
        if self.use_working_memory:
            if action_seq is not None and action_seq.shape[1] > 0:
                T = min(action_seq.shape[1], self.working_memory.memory_len)
                obs_for_memory = obs_feature_seq[:, -T:, :]
                action_for_memory = action_seq[:, -T:, :]
                working_mem = self.working_memory(obs_for_memory, action_for_memory)
            else:
                # 推理时没有历史动作：用当前 obs 构造虚拟历史
                # 这样和训练时的分布更接近（而不是用零向量）
                T = min(2, self.working_memory.memory_len)  # 用 2 步虚拟历史
                obs_for_memory = current_obs_feature.unsqueeze(1).expand(-1, T, -1)
                # 用零动作（或当前 obs 对应的典型动作）
                dummy_actions = torch.zeros(B, T, self.action_dim, device=current_obs_feature.device)
                working_mem = self.working_memory(obs_for_memory, dummy_actions)
            cond.append(working_mem)

        # L2 Episodic Memory
        if self.use_episodic_memory:
            if dino_feature_seq is not None and dino_feature_seq.shape[1] > 1:
                # Training mode: build memory bank from sequence
                memory_bank = self.build_episodic_memory_bank(dino_feature_seq)
                current_dino_feature = dino_feature_seq[:, -1, :]
                episodic_mem = self.episodic_memory(current_dino_feature, memory_bank)
            elif self.episodic_memory_bank is not None:
                # Inference mode: use accumulated memory bank
                current_dino_feature = dino_feature_seq[:, -1, :] if dino_feature_seq is not None else None
                if current_dino_feature is None:
                    # Fallback: use last stored feature or zeros
                    if self.prev_dino_feature is not None:
                        current_dino_feature = self.prev_dino_feature
                    else:
                        current_dino_feature = torch.zeros(B, self.dino_feature_dim, device=current_obs_feature.device)
                episodic_mem = self.episodic_memory(current_dino_feature, self.episodic_memory_bank)
            else:
                # No memory yet
                episodic_mem = torch.zeros(B, self.memory_dim, device=current_obs_feature.device)
            cond.append(episodic_mem)

        return torch.cat(cond, dim=-1)

    def forward(self, data, train_mode=True):
        """
        Training forward: Uses the SAME memory logic as inference.
        """
        _ = train_mode
        data = self.preprocess_input(data, train_mode=True)

        # Encode observation sequence (same as inference)
        obs_feature_seq = self.encode_obs_sequence(data)  # (B, T, final_obs_dim)

        # Extract DINOv2 features (same as inference)
        dino_feature_seq = self.extract_dino_features_from_data(data)

        # Get action sequence for working memory
        # Use the part of the sequence that corresponds to observations
        T_obs = obs_feature_seq.shape[1]
        action_seq = data["actions"][:, :T_obs, :]  # (B, T, action_dim)

        # Compute memory condition using unified function
        cond = self.compute_memory_condition(obs_feature_seq, action_seq, dino_feature_seq)

        # Flow Matching loss
        start_idx = self.n_obs_steps - 1
        action = data["actions"][:, start_idx : start_idx + self.horizon, :]
        B = action.shape[0]

        t = torch.rand(B, device=action.device)
        x_0 = torch.randn_like(action)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * action
        v_target = action - x_0
        v_pred = self.vector_field_net(x_t, t, global_cond=cond)

        return {"v_target": v_target, "v_pred": v_pred}

    def compute_loss(self, data, reduction="mean"):
        output = self.forward(data, train_mode=True)
        return F.mse_loss(output["v_pred"], output["v_target"], reduction=reduction)

    def update_episodic_memory_inference(self, dino_feature):
        """
        Update episodic memory bank during inference (single frame at a time).

        This is called each step during inference to accumulate events.

        Args:
            dino_feature: (B, D) current DINOv2 feature
        """
        if not self.use_episodic_memory or dino_feature is None:
            return

        # Detect event using single-frame comparison
        is_event, distance = self.dino_extractor.detect_event_single(
            dino_feature,
            self.prev_dino_feature,
            threshold_percentile=self.episodic_memory.event_threshold_percentile
        )

        if is_event:
            # Add to memory bank
            if self.episodic_memory_bank is None:
                self.episodic_memory_bank = dino_feature.unsqueeze(1)  # (B, 1, D)
            else:
                # Limit memory size
                if self.episodic_memory_bank.shape[1] >= self.episodic_memory.max_events:
                    self.episodic_memory_bank = self.episodic_memory_bank[:, 1:, :]
                self.episodic_memory_bank = torch.cat([
                    self.episodic_memory_bank,
                    dino_feature.unsqueeze(1)
                ], dim=1)

        self.prev_dino_feature = dino_feature.detach()

    def get_action(self, data):
        """
        Inference: Uses the SAME memory logic as training.

        Maintains obs_feature_history for working memory and accumulates
        episodic memory bank over the episode.
        """
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            B = data["obs"][list(self.image_encoders.keys())[0]].shape[0]

            # 1. Maintain image observation history
            for img_name in self.image_encoders.keys():
                x = data["obs"][img_name]  # (B, 1, C, H, W)
                if img_name not in self.obs_history:
                    self.obs_history[img_name] = deque(maxlen=self.n_obs_steps)
                self.obs_history[img_name].append(x.clone())

            # 2. Build data with full history (padded if needed)
            history_data = {"obs": {}, "task_emb": data["task_emb"]}
            for img_name in self.image_encoders.keys():
                history_list = list(self.obs_history[img_name])
                if len(history_list) < self.n_obs_steps:
                    pad_len = self.n_obs_steps - len(history_list)
                    history_list = [history_list[0]] * pad_len + history_list
                history_data["obs"][img_name] = torch.cat(history_list, dim=1)

            # 3. Encode observation sequence
            obs_feature_seq = self.encode_obs_sequence(history_data)  # (B, T, final_obs_dim)

            # 4. Track obs feature history for working memory
            current_obs_feature = obs_feature_seq[:, -1, :]  # (B, final_obs_dim)
            self.obs_feature_history.append(current_obs_feature)

            # 5. Build padded obs feature sequence from history
            memory_len = self.working_memory.memory_len if self.use_working_memory else 1
            obs_feature_list = list(self.obs_feature_history)
            if len(obs_feature_list) < memory_len:
                pad_len = memory_len - len(obs_feature_list)
                obs_feature_list = [obs_feature_list[0]] * pad_len + obs_feature_list
            obs_feature_seq_for_memory = torch.stack(obs_feature_list, dim=1)  # (B, T, dim)

            # 6. Build action sequence from history
            action_list = list(self.action_history)
            if len(action_list) == 0:
                action_seq = None
            else:
                action_seq = torch.stack(action_list, dim=1)  # (B, T, action_dim)

            # 7. Update episodic memory
            dino_feature_seq_for_memory = None
            if self.use_episodic_memory and self.dino_extractor is not None:
                # Extract DINOv2 feature from current observation
                for img_name in self.image_encoders.keys():
                    if "rgb" in img_name:
                        x = data["obs"][img_name]  # (B, 1, C, H, W)
                        current_dino = self.dino_extractor.extract_features(x.squeeze(1))  # (B, D)
                        break
                self.update_episodic_memory_inference(current_dino)

            # 8. Compute memory condition
            if self.use_working_memory:
                # Use only the part of obs history aligned with action history
                if action_seq is not None:
                    T_actions = action_seq.shape[1]
                    obs_for_memory = obs_feature_seq_for_memory[:, -T_actions:, :]
                else:
                    # No action history yet, pass None to signal no history
                    obs_for_memory = obs_feature_seq_for_memory
            else:
                obs_for_memory = obs_feature_seq

            cond = self.compute_memory_condition(
                obs_for_memory, action_seq, dino_feature_seq_for_memory
            )

            # 9. Flow Matching: Euler integration
            x = torch.randn(B, self.horizon, self.action_dim, device=obs_feature_seq.device)
            dt = 1.0 / self.num_integration_steps
            for i in range(self.num_integration_steps):
                t = i / self.num_integration_steps
                t_batch = torch.full((B,), t, device=obs_feature_seq.device, dtype=torch.float32)
                v = self.vector_field_net(x, t_batch, global_cond=cond)
                x = x + v * dt

            action = x[:, 0, :].detach()
            self.action_history.append(action)

        return action.cpu().numpy()

    def reset(self):
        """Clear all history buffers."""
        self.obs_history = {}
        self.obs_feature_history = deque(maxlen=16)
        self.action_history = deque(maxlen=16)
        self.episodic_memory_bank = None
        self.prev_dino_feature = None
        self.dino_feature_history = deque(maxlen=100)
        if self.dino_extractor is not None:
            self.dino_extractor.distance_history.clear()
            self.dino_extractor.event_threshold = None
