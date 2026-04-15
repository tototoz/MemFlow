"""
Memory-Augmented Flow Matching Policy (MemFlow).

Hierarchical memory architecture for long-horizon robot manipulation:
- L1 Action Memory: GRU over recent action trajectory (what I did)
- L2 Episodic Memory: Cross-attention over DINOv2 feature window (what happened)
- Adaptive Step Router: Memory-guided integration step selection (how carefully to act)

Key insight: LSTM encodes visual history (what I see), while L1 and L2 provide
complementary information — action dynamics and semantic task progress — that
the LSTM does not capture. The router leverages all three signals to choose
integration steps, unifying quality and efficiency in a single framework.

Reference:
- Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
- MemFlow: Memory-Augmented Flow Policy for Long-Horizon Manipulation

════════════════════════════════════════════════════════════════════════
配置参数（memflow_policy.yaml）
════════════════════════════════════════════════════════════════════════
  n_obs_steps       = 32    观测历史帧数（也是 DINOv2 窗口大小）
  horizon           = 8     预测动作步数
  action_memory_len = 16    Action Memory 最大历史长度

════════════════════════════════════════════════════════════════════════
训练数据流  forward()
════════════════════════════════════════════════════════════════════════

  数据加载（SequenceDataset, seq_len=39）
  ┌──────────────────────────────────────────────────────────────────┐
  │ data["obs"]["agentview_rgb"]   (B, 39, 3, 128, 128)             │
  │ data["obs"]["eye_in_hand_rgb"] (B, 39, 3, 128, 128)             │
  │ data["obs"]["joint_states"]    (B, 39, 7)                       │
  │ data["obs"]["gripper_states"]  (B, 39, 2)                       │
  │ data["task_emb"]               (B, 768)  ← BERT 句子嵌入        │
  │ data["actions"]                (B, 39, 7)                       │
  └──────────────────────────────────────────────────────────────────┘

  ── 阶段1：观测编码  encode_obs_sequence() ──────────────────────────
  agentview_rgb  (B,32,3,H,W) ─→ ResNet+语言调制 ─→ (B,32, 64)  ┐
  eye_in_hand    (B,32,3,H,W) ─→ ResNet+语言调制 ─→ (B,32, 64)  ├─ cat
  joint+gripper  (B,32, 9)    ─→ MLP            ─→ (B,32, 16)  ┤
  task_emb       (B,768)      ─→ MLP            ─→ (B,  64)    ┘
                                                 expand (B,32, 64)
                                                        ↓
                                               concat → (B, 32, 208)
                                                        ↓
                                               2层 LSTM（隐层256）
                                                        ↓
                                           obs_feature_seq (B, 32, 256)

  current_obs = obs_feature_seq[:, -1, :]  → (B, 256)

  ── 阶段2：L1 Action Memory（GRU）──────────────────────────────────
  action_history (B, T, 7)  T ≤ 16
         ↓ Linear(7 → 128)
         ↓ GRU(128, hidden=256)
         ↓ Linear(256 → 256)
  action_mem (B, 256)

  原理：LSTM 编码视觉历史（what I see），Action Memory 编码动作
  轨迹（what I did）。两者信息互补，无冗余。

  ── 阶段3：L2 Episodic Memory（全帧 Cross-Attention）────────────────
  agentview_rgb (B,32,3,128,128) → DINOv2 冻结 → dino_seq (B,32,384)
  query = Linear(384→256)(dino_seq[:,-1,:])          → (B,1,256)
  key   = Linear(384→256)(dino_seq)                  → (B,32,256)
  value = Linear(384→256)(dino_seq)                  → (B,32,256)
  CrossAttention → Linear(256→256)                   → (B, 256)

  原理：不做显式事件检测。Cross-attention 的 softmax 权重自动聚焦
  于语义变化大的帧（隐式关键帧选择），end-to-end 可学习。

  ── 阶段4：条件向量 + 自适应步数路由 ────────────────────────────────
  cond = cat([current_obs, action_mem, episodic_mem]) → (B, 768)

  Router: MLP(768→256→128→4) → logits for {2,4,8,16} steps
  训练 Phase1: 不用 Router，固定步数
  训练 Phase2: Gumbel-Softmax 选步数
  推理: argmax 选步数

  ── 阶段5：Flow Matching 损失 ────────────────────────────────────────
  target_action = actions[:, 31:39, :]       (B, 8, 7)
  标准 Conditional Flow Matching loss (同前)

════════════════════════════════════════════════════════════════════════
推理数据流  get_action()    每步环境返回 1 帧
════════════════════════════════════════════════════════════════════════

  实例缓冲区（跨步持久）：
    obs_history[img_name]   deque(maxlen=32)  ← 原始图像帧
    action_history          deque(maxlen=16)  ← 已执行的动作
    dino_feature_buffer     deque(maxlen=32)  ← DINOv2 特征窗口

  每步推理流程：

  step t  输入: obs (B,1,3,H,W),  task_emb (B,768)
    │
    ├─ 1. obs_history.append(obs)，不足32帧时重复第一帧填充
    ├─ 2. encode_obs_sequence(history_data) → obs_feature_seq (B,32,256)
    ├─ 3. DINOv2 提取当前帧 → dino_feature_buffer.append
    ├─ 4. action_seq = stack(action_history) 或 None
    ├─ 5. compute_memory_condition() → cond (B,768)
    ├─ 6. Router 选步数 n（或固定步数）
    ├─ 7. Euler 积分（n步）生成动作
    └─ 8. action_history.append(action)

  张量维度速查：
    obs_feature_seq         (B, 32, 256)
    dino_feature_seq        (B, 32, 384)  训练 / (B, ≤32, 384)  推理
    action_mem              (B, 256)
    episodic_mem            (B, 256)
    cond                    (B, 768)      = 256+256+256
    router_logits           (B, 4)        → {2,4,8,16}
    v_pred / v_target       (B,   8,   7)
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
# DINOv2 Feature Extractor
# ============================================================

class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 feature extractor for semantic visual features."""

    def __init__(self, model_name="dinov2_small", freeze=True, local_path=None):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze

        try:
            from transformers import AutoModel

            if local_path is not None and os.path.exists(local_path):
                print(f"[DINOv2] Loading from local: {local_path}")
                self.model = AutoModel.from_pretrained(local_path)
            elif os.path.exists(model_name):
                print(f"[DINOv2] Loading from local: {model_name}")
                self.model = AutoModel.from_pretrained(model_name)
            else:
                hf_model_name = model_name.replace("_", "-")
                print(f"[DINOv2] Downloading: facebook/{hf_model_name}")
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

    def extract_features(self, image):
        """Extract CLS token features. image: (B, C, H, W) in [0, 1]."""
        _, _, H, W = image.shape
        if H != 224 or W != 224:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

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
        """Extract features for a sequence. images: (B, T, C, H, W) → (B, T, D)."""
        B, T, C, H, W = images.shape
        images = images.reshape(B * T, C, H, W)
        features = self.extract_features(images)
        return features.reshape(B, T, -1)


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
# L1 Action Memory: GRU over action trajectory
# ============================================================

class ActionMemory(nn.Module):
    """
    L1 Action Memory for short-term motion coherence.

    Encodes the recent action trajectory (what I did) using a GRU.
    Complementary to the LSTM obs encoder (what I see) — no redundancy.
    """
    def __init__(self, action_dim=7, hidden_dim=128, gru_dim=256, output_dim=256):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(gru_dim, output_dim)

    def forward(self, actions):
        """
        Args:
            actions: (B, T, action_dim)  T can be 1..action_memory_len
        Returns:
            (B, output_dim)
        """
        x = self.action_proj(actions)           # (B, T, hidden_dim)
        _, h_n = self.gru(x)                    # h_n: (1, B, gru_dim)
        return self.output_proj(h_n.squeeze(0))  # (B, output_dim)


# ============================================================
# L2 Episodic Memory: Full-frame DINOv2 Cross-Attention
# ============================================================

class EpisodicMemory(nn.Module):
    """
    L2 Episodic Memory for mid-term task progress tracking.

    Uses cross-attention over the full DINOv2 feature window.
    The softmax attention weights serve as implicit key-frame selection —
    frames with large semantic changes naturally receive higher weights.
    """
    def __init__(self, dino_feature_dim=384, hidden_dim=256, output_dim=256, n_heads=4):
        super().__init__()
        self.dino_feature_dim = dino_feature_dim
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(dino_feature_dim, hidden_dim)
        self.key_proj = nn.Linear(dino_feature_dim, hidden_dim)
        self.value_proj = nn.Linear(dino_feature_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, current_dino, dino_history):
        """
        Args:
            current_dino: (B, D) current frame DINOv2 feature (query)
            dino_history: (B, T, D) DINOv2 feature window (keys/values)
        Returns:
            (B, output_dim)
        """
        B = current_dino.shape[0]

        if dino_history is None or dino_history.shape[1] == 0:
            return torch.zeros(B, self.output_proj.out_features, device=current_dino.device)

        query = self.query_proj(current_dino).unsqueeze(1)   # (B, 1, hidden)
        keys = self.key_proj(dino_history)                    # (B, T, hidden)
        values = self.value_proj(dino_history)                # (B, T, hidden)

        attended, _ = self.cross_attn(query, keys, values)
        return self.output_proj(attended.squeeze(1))          # (B, output_dim)


# ============================================================
# Adaptive Step Router
# ============================================================

class AdaptiveStepRouter(nn.Module):
    """
    Memory-guided adaptive integration step router.

    Uses the full condition vector (obs + action memory + episodic memory)
    to predict how many Euler integration steps are needed.

    Training: Gumbel-Softmax for differentiable discrete selection.
    Inference: argmax.
    """
    def __init__(self, cond_dim=768, hidden_dims=(256, 128),
                 step_options=(2, 4, 8, 16), gumbel_tau=1.0):
        super().__init__()
        self.step_options = step_options
        self.gumbel_tau = gumbel_tau
        n_options = len(step_options)

        layers = []
        in_dim = cond_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_options))
        self.net = nn.Sequential(*layers)

    def forward(self, cond, training=False):
        """
        Args:
            cond: (B, cond_dim) condition vector
            training: if True, use Gumbel-Softmax; else argmax
        Returns:
            step_weights: (B, n_options) soft weights (training) or one-hot (inference)
            logits: (B, n_options) raw logits
        """
        logits = self.net(cond)  # (B, n_options)

        if training:
            step_weights = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        else:
            idx = logits.argmax(dim=-1)
            step_weights = F.one_hot(idx, num_classes=len(self.step_options)).float()

        return step_weights, logits

    def get_num_steps(self, cond):
        """Inference helper: returns integer step count."""
        with torch.no_grad():
            logits = self.net(cond)
            idx = logits.argmax(dim=-1)  # (B,)
            return self.step_options[idx[0].item()]


# ============================================================
# MemFlow Policy
# ============================================================

class MemFlowPolicy(BasePolicy):
    """
    Memory-Augmented Flow Matching Policy.

    - L1 Action Memory: GRU over recent action trajectory (what I did)
    - L2 Episodic Memory: Cross-attention over DINOv2 feature window (what happened)
    - Adaptive Step Router: Memory-guided integration step selection
    """
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # Flow Matching parameters
        self.num_integration_steps = policy_cfg.get("num_integration_steps", 10)
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps

        # Memory toggles
        self.use_action_memory = policy_cfg.get("use_working_memory", False)
        self.use_episodic_memory = policy_cfg.get("use_episodic_memory", False)
        self.use_adaptive_router = policy_cfg.get("use_adaptive_router", False)

        # Action memory length
        self.action_memory_len = policy_cfg.get("working_memory_len", 16)

        # Episodic memory window (same as n_obs_steps by default)
        self.episodic_window = policy_cfg.get("episodic_memory_window", self.n_obs_steps)

        # Verify alignment
        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        # Inference buffers
        self.obs_history = {}
        self.action_history = deque(maxlen=self.action_memory_len)
        self.dino_feature_buffer = deque(maxlen=self.episodic_window)

        # ---- DINOv2 Feature Extractor ----
        if self.use_episodic_memory:
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
        self.action_dim = shape_meta["ac_dim"]

        # ---- Memory Modules ----
        memory_dim = policy_cfg.get("memory_dim", 256)
        self.memory_dim = memory_dim

        if self.use_action_memory:
            self.action_memory = ActionMemory(
                action_dim=self.action_dim,
                hidden_dim=policy_cfg.get("action_memory_hidden_dim", 128),
                gru_dim=policy_cfg.get("action_memory_gru_dim", 256),
                output_dim=memory_dim,
            )
            print("[MemFlow] L1 Action Memory enabled (GRU)")

        if self.use_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                dino_feature_dim=self.dino_feature_dim,
                hidden_dim=policy_cfg.get("episodic_memory_hidden_dim", 256),
                output_dim=memory_dim,
                n_heads=policy_cfg.get("episodic_memory_n_heads", 4),
            )
            print(f"[MemFlow] L2 Episodic Memory enabled (window={self.episodic_window})")

        # ---- Vector Field Network ----
        cond_dim = self.final_obs_dim
        if self.use_action_memory:
            cond_dim += memory_dim
        if self.use_episodic_memory:
            cond_dim += memory_dim
        self.cond_dim = cond_dim

        self.vector_field_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=cond_dim,
            time_embed_dim=policy_cfg.get("time_embed_dim", 256),
            down_dims=policy_cfg.down_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
            cond_predict_scale=policy_cfg.cond_predict_scale,
        )

        # ---- Adaptive Step Router ----
        if self.use_adaptive_router:
            step_options = tuple(policy_cfg.get("router_step_options", [2, 4, 8, 16]))
            self.router = AdaptiveStepRouter(
                cond_dim=cond_dim,
                hidden_dims=tuple(policy_cfg.get("router_hidden_dims", [256, 128])),
                step_options=step_options,
                gumbel_tau=policy_cfg.get("router_gumbel_tau", 1.0),
            )
            self.router_lambda_speed = policy_cfg.get("router_lambda_speed", 0.1)
            print(f"[MemFlow] Adaptive Step Router enabled, options={step_options}")

        self.to(self.device)
        if self.dino_extractor is not None:
            self.dino_extractor.to(self.device)

    def encode_obs_sequence(self, data):
        """Encode observation sequence to feature sequence. (B, T, final_obs_dim)"""
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

        extra = self.extra_encoder(data["obs"])
        if extra.shape[1] == 1 and T_obs > 1:
            extra = extra.expand(-1, T_obs, -1)
        encoded.append(extra)

        lang_h = self.language_encoder(data)
        encoded.append(lang_h.unsqueeze(1).expand(-1, T_obs, -1))

        x = torch.cat(encoded, dim=-1)
        output, _ = self.obs_temporal_encoder(x)
        return output

    def extract_dino_features_from_data(self, data):
        """Extract DINOv2 features from observation data. Returns (B, T, D) or None."""
        if self.dino_extractor is None:
            return None
        for img_name in self.image_encoders.keys():
            if "rgb" in img_name:
                x = data["obs"][img_name]
                return self.dino_extractor.extract_features_batch(x)
        return None

    def compute_memory_condition(self, obs_feature_seq, action_seq, dino_feature_seq):
        """
        Compute memory-augmented condition vector.
        Unified function for BOTH training and inference.
        """
        B = obs_feature_seq.shape[0]
        current_obs = obs_feature_seq[:, -1, :]
        cond = [current_obs]

        # L1 Action Memory
        if self.use_action_memory:
            if action_seq is not None and action_seq.shape[1] > 0:
                action_mem = self.action_memory(action_seq)
            else:
                action_mem = torch.zeros(B, self.memory_dim, device=current_obs.device)
            cond.append(action_mem)

        # L2 Episodic Memory
        if self.use_episodic_memory:
            if dino_feature_seq is not None and dino_feature_seq.shape[1] > 0:
                current_dino = dino_feature_seq[:, -1, :]
                episodic_mem = self.episodic_memory(current_dino, dino_feature_seq)
            else:
                episodic_mem = torch.zeros(B, self.memory_dim, device=current_obs.device)
            cond.append(episodic_mem)

        return torch.cat(cond, dim=-1)

    def forward(self, data, train_mode=True):
        """
        Training forward pass.

        - L1 uses action history from the dataset
        - L2 uses full DINOv2 feature window (n_obs_steps frames)
        - Training augmentation: randomly truncates action/DINOv2 history
        """
        _ = train_mode
        data = self.preprocess_input(data, train_mode=True)

        B = data["obs"][list(self.image_encoders.keys())[0]].shape[0]
        device = data["obs"][list(self.image_encoders.keys())[0]].device

        obs_feature_seq = self.encode_obs_sequence(data)
        dino_feature_seq = self.extract_dino_features_from_data(data)
        actions = data["actions"]

        # Action history for L1
        n_history = self.n_obs_steps - 1
        action_seq = actions[:, :n_history, :] if n_history > 0 else None

        # Training augmentation: randomly truncate to simulate inference warm-up
        if self.training and self.use_action_memory and action_seq is not None:
            if torch.rand(1).item() < 0.3:
                T_keep = torch.randint(0, min(n_history, self.action_memory_len) + 1, (1,)).item()
                action_seq = action_seq[:, -T_keep:, :] if T_keep > 0 else None

        if self.training and self.use_episodic_memory and dino_feature_seq is not None:
            if torch.rand(1).item() < 0.2:
                T_dino = dino_feature_seq.shape[1]
                T_keep = torch.randint(1, T_dino + 1, (1,)).item()
                dino_feature_seq = dino_feature_seq[:, -T_keep:, :]

        # Compute condition vector
        cond = self.compute_memory_condition(obs_feature_seq, action_seq, dino_feature_seq)

        # Flow Matching loss
        target_action = actions[:, n_history:n_history + self.horizon, :]
        time_t = torch.rand(B, device=device)
        x_0 = torch.randn_like(target_action)
        x_t = (1 - time_t.view(-1, 1, 1)) * x_0 + time_t.view(-1, 1, 1) * target_action
        v_target = target_action - x_0
        v_pred = self.vector_field_net(x_t, time_t, global_cond=cond)

        result = {"v_target": v_target, "v_pred": v_pred}

        # Router loss (Phase 2 training)
        if self.use_adaptive_router and self.training:
            step_weights, logits = self.router(cond.detach(), training=True)
            step_options = torch.tensor(self.router.step_options, device=device, dtype=torch.float32)
            # Speed loss: weighted average of step counts (encourage fewer steps)
            avg_steps = (step_weights * step_options).sum(dim=-1).mean()
            result["router_speed_loss"] = avg_steps / step_options.max()
            result["router_logits"] = logits

        return result

    def compute_loss(self, data, reduction="mean"):
        output = self.forward(data, train_mode=True)
        flow_loss = F.mse_loss(output["v_pred"], output["v_target"], reduction=reduction)

        if "router_speed_loss" in output:
            return flow_loss + self.router_lambda_speed * output["router_speed_loss"]
        return flow_loss

    def get_action(self, data):
        """Inference: generate action with memory-augmented condition."""
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            B = data["obs"][list(self.image_encoders.keys())[0]].shape[0]

            # 1. Maintain image observation history
            for img_name in self.image_encoders.keys():
                x = data["obs"][img_name]
                if img_name not in self.obs_history:
                    self.obs_history[img_name] = deque(maxlen=self.n_obs_steps)
                self.obs_history[img_name].append(x.clone())

            # 2. Build padded history data
            history_data = {"obs": {}, "task_emb": data["task_emb"]}
            for img_name in self.image_encoders.keys():
                history_list = list(self.obs_history[img_name])
                if len(history_list) < self.n_obs_steps:
                    pad_len = self.n_obs_steps - len(history_list)
                    history_list = [history_list[0]] * pad_len + history_list
                history_data["obs"][img_name] = torch.cat(history_list, dim=1)

            for key in data["obs"].keys():
                if key not in self.image_encoders:
                    history_data["obs"][key] = data["obs"][key]

            # 3. Encode observations
            obs_feature_seq = self.encode_obs_sequence(history_data)

            # 4. Build action sequence from history
            action_list = list(self.action_history)
            action_seq = torch.stack(action_list, dim=1) if len(action_list) > 0 else None

            # 5. Update DINOv2 feature buffer
            dino_feature_seq = None
            if self.use_episodic_memory and self.dino_extractor is not None:
                for img_name in self.image_encoders.keys():
                    if "rgb" in img_name:
                        current_dino = self.dino_extractor.extract_features(
                            data["obs"][img_name].squeeze(1)
                        )
                        self.dino_feature_buffer.append(current_dino.detach())
                        dino_feature_seq = torch.stack(
                            list(self.dino_feature_buffer), dim=1
                        )
                        break

            # 6. Compute condition
            cond = self.compute_memory_condition(obs_feature_seq, action_seq, dino_feature_seq)

            # 7. Determine integration steps
            if self.use_adaptive_router:
                n_steps = self.router.get_num_steps(cond)
            else:
                n_steps = self.num_integration_steps

            # 8. Midpoint Euler integration
            x = torch.randn(B, self.horizon, self.action_dim, device=cond.device)
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = (i + 0.5) / n_steps
                t_batch = torch.full((B,), t, device=cond.device, dtype=torch.float32)
                v = self.vector_field_net(x, t_batch, global_cond=cond)
                x = x + v * dt

            action = x[:, 0, :].detach()
            self.action_history.append(action)

        return action.cpu().numpy()

    def reset(self):
        """Clear all history buffers."""
        self.obs_history = {}
        self.action_history = deque(maxlen=self.action_memory_len)
        self.dino_feature_buffer = deque(maxlen=self.episodic_window)
