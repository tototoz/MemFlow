"""
Flow Matching Policy for Robot Manipulation.

Flow Matching is an alternative to Diffusion that learns a continuous-time
vector field to transport samples from noise to data distribution.

Key differences from Diffusion Policy:
- Training: Learn vector field v_t(x_t, t) instead of noise prediction
- Inference: ODE integration (Euler method) instead of iterative denoising
- More efficient: Can achieve good results with fewer integration steps

Reference:
- Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)
- Conditional Flow Matching (Tong et al., ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from einops.layers.torch import Rearrange

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *


# ============================================================
# UNet building blocks (same as Diffusion Policy)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time t in [0, 1]."""
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
    """1D residual block with FiLM conditioning."""
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
    """
    1D U-Net for Flow Matching.
    Predicts vector field v_t(x_t, t) instead of noise.

    Args:
        input_dim: action dimension
        global_cond_dim: observation feature dimension
        time_embed_dim: time embedding size
        down_dims: channel sizes at each U-Net level
        kernel_size: conv kernel size
        n_groups: GroupNorm groups
        cond_predict_scale: FiLM scale+bias (True) or bias only (False)
    """
    def __init__(self, input_dim, global_cond_dim=None, time_embed_dim=256,
                 down_dims=(128, 256, 512), kernel_size=3, n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Time encoder for t in [0, 1]
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        cond_dim = time_embed_dim + (global_cond_dim or 0)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Down blocks
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # Mid block
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        # Up blocks
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
        """
        Args:
            sample: (B, T, input_dim) current state x_t
            t: (B,) time in [0, 1]
            global_cond: (B, global_cond_dim) observation condition
        Returns:
            (B, T, input_dim) predicted vector field v_t
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        x = einops.rearrange(sample, 'b h t -> b t h')

        # Time embedding
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32, device=x.device)
        elif t.ndim == 0:
            t = t[None].to(x.device)
        t = t.expand(x.shape[0])

        global_feature = self.time_encoder(t)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Down path
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Mid
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Up path
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # (B, D, T) -> (B, T, D)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


# ============================================================
# Flow Matching Policy
# ============================================================

class FlowMatchingPolicy(BasePolicy):
    """
    Flow Matching Policy for robot manipulation.

    Uses Conditional Flow Matching (CFM) with optimal transport path:
    - x_t = (1 - t) * x_0 + t * x_1, where x_0 ~ N(0, I), x_1 = data
    - v_t = x_1 - x_0 (ground truth vector field)
    - Train: predict v_t given x_t and t

    Inference uses Euler integration to transport noise to data.
    """
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # Flow Matching hyperparameters
        self.num_integration_steps = policy_cfg.get("num_integration_steps", 10)
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps

        # Verify alignment
        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        # History buffer for inference
        self.obs_history = {}

        # ---- Observation encoders (same as Diffusion Policy) ----
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

        # ---- Vector Field Network ----
        self.vector_field_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.final_obs_dim,
            time_embed_dim=policy_cfg.get("time_embed_dim", 256),
            down_dims=policy_cfg.down_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
            cond_predict_scale=policy_cfg.cond_predict_scale,
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
                from collections import deque
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
            from collections import deque
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
        Training forward: Conditional Flow Matching loss.

        For optimal transport path:
        - x_t = (1 - t) * x_0 + t * x_1
        - v_t = x_1 - x_0 (target vector field)
        - Loss: ||v_theta(x_t, t) - v_t||^2
        """
        data = self.preprocess_input(data, train_mode=True)
        obs_cond = self.encode_obs(data)

        # Action target
        start_idx = self.n_obs_steps - 1
        action = data["actions"][:, start_idx : start_idx + self.horizon, :]
        B = action.shape[0]

        # Sample random t in [0, 1]
        t = torch.rand(B, device=action.device)

        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(action)

        # Optimal transport interpolation: x_t = (1 - t) * x_0 + t * x_1
        # where x_1 = action (data)
        x_t = (1 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * action

        # Target vector field: v_t = x_1 - x_0 = action - noise
        v_target = action - x_0

        # Predicted vector field
        v_pred = self.vector_field_net(x_t, t, global_cond=obs_cond)

        return {"v_target": v_target, "v_pred": v_pred}

    def compute_loss(self, data, reduction="mean"):
        output = self.forward(data, train_mode=True)
        return F.mse_loss(output["v_pred"], output["v_target"], reduction=reduction)

    def get_action(self, data):
        """
        Inference: Euler integration to transport noise to data.

        Starting from x_0 ~ N(0, I), integrate:
        x_{t+dt} = x_t + v_theta(x_t, t) * dt

        With N steps: dt = 1/N
        """
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            obs_cond = self.encode_obs(data)
            B = obs_cond.shape[0]

            # Start from noise
            x = torch.randn(B, self.horizon, self.action_dim, device=obs_cond.device)

            # Midpoint Euler integration
            dt = 1.0 / self.num_integration_steps
            for i in range(self.num_integration_steps):
                t = (i + 0.5) / self.num_integration_steps
                t_batch = torch.full((B,), t, device=obs_cond.device, dtype=torch.float32)
                v = self.vector_field_net(x, t_batch, global_cond=obs_cond)
                x = x + v * dt

        return x[:, 0, :].detach().cpu().numpy()

    def reset(self):
        """Clear observation history buffer."""
        self.obs_history = {}
