"""
MemFlow: Diffusion Policy with Denoising Convergence Monitor (DCM).

When dcm.enabled=false, behaves identically to DiffusionPolicy (DDPM, 100 steps).
When dcm.enabled=true, a lightweight convergence monitor runs alongside DDIM
sampling and triggers early stopping when the denoised estimate has converged.

Core idea: At each DDIM step, the model produces a0_hat — its current best
estimate of the clean action. For many inputs, this estimate stabilizes well
before the prescribed number of steps. The DCM monitors this convergence in
real time and halts denoising when further steps won't change the output.

Key advantages over discrete step selection:
  - Online: decides based on actual denoising trajectory, not just observation
  - Continuous: any step count from min_steps to K_max
  - Seed-aware: different noise initializations stop at different steps
  - Theoretically grounded: truncation error bound justifies early stopping

Training (end-to-end, single phase):
  dcm.enabled=false: Standard DDPM training only
  dcm.enabled=true:  Joint training: DDPM loss + lambda_dcm * DCM loss
  DCM labels are generated online from the current UNet's DDIM trajectory,
  so the monitor co-evolves with the UNet and generalizes across tasks.

Mathematical foundation:
  - Convergence quantity: Delta^(i) = ||a0_hat^(i) - a0_hat^(i-1)||^2 / (H*d_a)
  - Truncation bound: if Delta < eps for remaining steps, total error <= O((K-i*)^2 * eps)
  - Convergence rate depends on conditional distribution complexity (Theorem 1)
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
# Denoising Convergence Monitor (DCM)
# ============================================================

def build_dcm_features(a0_cur, a0_prev, eps_pred, obs_cond, progress, snr):
    """
    Build 12-dim feature vector for convergence monitoring.

    Args:
        a0_cur: (B, H, d_a) current denoised estimate
        a0_prev: (B, H, d_a) previous denoised estimate (or None)
        eps_pred: (B, H, d_a) noise prediction
        obs_cond: (B, d_c) observation condition
        progress: float, normalized step index i/K_max
        snr: float, alpha_bar at current timestep
    Returns:
        features: (B, 12) feature vector
    """
    B = a0_cur.shape[0]
    device = a0_cur.device
    a0_flat = a0_cur.reshape(B, -1)
    Hda = float(a0_flat.shape[1])

    f_mean = a0_flat.mean(dim=1, keepdim=True)
    f_std = a0_flat.std(dim=1, keepdim=True)
    f_min = a0_flat.min(dim=1, keepdim=True).values
    f_max = a0_flat.max(dim=1, keepdim=True).values

    if a0_prev is not None:
        diff = (a0_cur - a0_prev).reshape(B, -1)
        delta = (diff ** 2).mean(dim=1, keepdim=True)
        log_delta = torch.log(delta + 1e-8)
    else:
        delta = torch.ones(B, 1, device=device)
        log_delta = torch.zeros(B, 1, device=device)

    eps_flat = eps_pred.reshape(B, -1)
    eps_norm = eps_flat.norm(dim=1, keepdim=True) / (Hda ** 0.5)
    cond_norm = obs_cond.norm(dim=1, keepdim=True) / (obs_cond.shape[1] ** 0.5)

    progress_t = torch.full((B, 1), progress, device=device)
    snr_t = torch.full((B, 1), snr, device=device)

    # delta_ratio: contraction rate (clamped for stability)
    # Will be set by caller when tracking previous delta
    delta_ratio = torch.ones(B, 1, device=device)

    features = torch.cat([
        f_mean, f_std, f_min, f_max,
        delta, log_delta, delta_ratio,
        progress_t, snr_t,
        cond_norm, eps_norm,
        torch.full((B, 1), Hda, device=device),
    ], dim=1)

    return features


class ConvergenceMonitor(nn.Module):
    """
    Lightweight network that predicts whether denoising has converged.

    Condition-aware variant (DCM-CA): projects obs_cond to 16-dim and
    concatenates with the 12-dim trajectory features.

    ~8K parameters, runs inside the DDIM loop at negligible cost.
    """
    def __init__(self, feature_dim=12, cond_dim=256, cond_proj_dim=16,
                 hidden_dims=(64, 32), use_cond=True):
        super().__init__()
        self.use_cond = use_cond

        if use_cond:
            self.cond_proj = nn.Linear(cond_dim, cond_proj_dim)
            input_dim = feature_dim + cond_proj_dim
        else:
            self.cond_proj = None
            input_dim = feature_dim

        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features, obs_cond=None):
        """
        Args:
            features: (B, 12) trajectory features from build_dcm_features
            obs_cond: (B, cond_dim) observation condition (if use_cond)
        Returns:
            h: (B, 1) convergence score in [0, 1]
        """
        if self.use_cond and obs_cond is not None:
            cond_feat = self.cond_proj(obs_cond)
            x = torch.cat([features, cond_feat], dim=1)
        else:
            x = features
        return torch.sigmoid(self.net(x))


# ============================================================
# UNet building blocks (official Diffusion Policy architecture)
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
    def __init__(self, input_dim, global_cond_dim=None, diffusion_step_embed_dim=256,
                 down_dims=(128, 256, 512), kernel_size=3, n_groups=8,
                 cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + (global_cond_dim or 0)

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

    def forward(self, sample, timestep, global_cond=None):
        x = einops.rearrange(sample, 'b h t -> b t h')

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
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
# MemFlowPolicy
# ============================================================

class MemFlowPolicy(BasePolicy):
    """
    Diffusion Policy with optional Denoising Convergence Monitor.

    dcm.enabled=false: Standard DDPM (identical to DiffusionPolicy).
    dcm.enabled=true:  DDIM sampling with online convergence-based early stopping.
    """
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        self.num_diffusion_iters = policy_cfg.num_diffusion_iters
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps

        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        self.obs_history = {}

        # ---- DDPM noise schedule ----
        betas = torch.linspace(policy_cfg.beta_start, policy_cfg.beta_end,
                               self.num_diffusion_iters)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.cat([posterior_variance[1:2],
                                                  posterior_variance[1:]])))
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alphas_cumprod_prev) /
                             (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1.0 - alphas_cumprod))

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

        # ---- Denoising U-Net ----
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.final_obs_dim,
            diffusion_step_embed_dim=policy_cfg.diffusion_step_embed_dim,
            down_dims=policy_cfg.down_dims,
            kernel_size=policy_cfg.kernel_size,
            n_groups=policy_cfg.n_groups,
            cond_predict_scale=policy_cfg.cond_predict_scale,
        )

        # ---- Denoising Convergence Monitor (DCM) ----
        dcm_cfg = policy_cfg.get("dcm", {})
        self.dcm_enabled = dcm_cfg.get("enabled", False)

        if self.dcm_enabled:
            self.dcm_K_max = dcm_cfg.get("K_max", 50)
            self.dcm_K_train = dcm_cfg.get("K_train", 20)
            self.dcm_min_steps = dcm_cfg.get("min_steps", 5)
            self.dcm_threshold = dcm_cfg.get("threshold", 0.7)
            self.dcm_epsilon = dcm_cfg.get("epsilon", 0.001)
            self.dcm_lambda = dcm_cfg.get("lambda_dcm", 0.1)

            self.convergence_monitor = ConvergenceMonitor(
                feature_dim=12,
                cond_dim=self.final_obs_dim,
                cond_proj_dim=16,
                hidden_dims=tuple(dcm_cfg.get("hidden_dims", [64, 32])),
                use_cond=dcm_cfg.get("use_cond", True),
            )

            T = self.num_diffusion_iters
            K = min(self.dcm_K_max, T)
            tau = torch.linspace(0, T - 1, K, dtype=torch.long)
            self.register_buffer('ddim_tau', tau)

            K_train = min(self.dcm_K_train, T)
            tau_train = torch.linspace(0, T - 1, K_train, dtype=torch.long)
            self.register_buffer('ddim_tau_train', tau_train)

            n_params = sum(p.numel() for p in self.convergence_monitor.parameters())
            print(f"[MemFlow] DCM enabled: K_max={K}, K_train={K_train}, "
                  f"lambda={self.dcm_lambda}, params={n_params}")

        self.to(self.device)

    def encode_obs(self, data):
        """Encode observations to a condition vector (B, obs_hidden_size)."""
        encoded = []

        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape

            if T == 1:
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
                x = x[:, :self.n_obs_steps, :, :, :]
                T = self.n_obs_steps

            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"].reshape(B, 1, -1).repeat(1, T, 1).reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

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
        """Phase 1 training: standard DDPM noise prediction (DCM not involved)."""
        data = self.preprocess_input(data, train_mode=True)
        obs_cond = self.encode_obs(data)

        start_idx = self.n_obs_steps - 1
        action = data["actions"][:, start_idx : start_idx + self.horizon, :]
        B = action.shape[0]

        t = torch.randint(0, self.num_diffusion_iters, (B,), device=action.device)
        noise = torch.randn_like(action)

        noisy_action = (
            self.sqrt_alphas_cumprod[t, None, None] * action +
            self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        )

        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs_cond)
        return {"noise": noise, "noise_pred": noise_pred, "obs_cond": obs_cond}

    def compute_loss(self, data, reduction="mean"):
        """
        Joint loss: DDPM noise prediction + DCM convergence monitor.

        When dcm_enabled: loss = ddpm_loss + lambda_dcm * dcm_loss
        When disabled: loss = ddpm_loss (identical to DiffusionPolicy)
        """
        output = self.forward(data, train_mode=True)
        ddpm_loss = F.mse_loss(output["noise_pred"], output["noise"], reduction=reduction)

        if not self.dcm_enabled or not self.training:
            return ddpm_loss

        obs_cond = output.get("obs_cond")
        if obs_cond is None:
            return ddpm_loss

        dcm_loss = self._compute_dcm_loss(obs_cond.detach())
        return ddpm_loss + self.dcm_lambda * dcm_loss

    def _compute_dcm_loss(self, obs_cond):
        """
        Online DCM training: run short DDIM, generate soft labels, train monitor.

        UNet runs under no_grad (DDIM trajectory is detached).
        Only the ConvergenceMonitor receives gradients.
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        tau = self.ddim_tau_train
        K = len(tau)
        epsilon = self.dcm_epsilon

        # Run DDIM and collect a0_hat trajectory (no grad for UNet)
        a0_list = []
        eps_list = []
        with torch.no_grad():
            action = torch.randn(B, self.horizon, self.action_dim, device=device)
            for step_idx in reversed(range(K)):
                t_cur = tau[step_idx]
                t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
                eps_pred = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

                alpha_bar_cur = self.alphas_cumprod[t_cur]
                a0_cur = (
                    (action - torch.sqrt(1.0 - alpha_bar_cur) * eps_pred)
                    / torch.sqrt(alpha_bar_cur)
                )

                a0_list.append(a0_cur)
                eps_list.append(eps_pred)

                if step_idx > 0:
                    alpha_bar_prev = self.alphas_cumprod[tau[step_idx - 1]]
                    action = (
                        torch.sqrt(alpha_bar_prev) * a0_cur
                        + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
                    )

        # a0_list[0] = first denoising step (high noise), a0_list[K-1] = last step
        # Compute deltas
        deltas = []
        Hda = float(self.horizon * self.action_dim)
        for i in range(1, K):
            diff = (a0_list[i] - a0_list[i - 1]).reshape(B, -1)
            deltas.append((diff ** 2).mean(dim=1))  # (B,)

        if len(deltas) == 0:
            return torch.tensor(0.0, device=device)

        deltas_t = torch.stack(deltas, dim=0)  # (K-1, B)

        # Generate soft labels: y^{(i)} = sigma(lam * (eps - max_future_delta) / eps)
        lam = 15.0
        soft_labels = []
        for i in range(len(deltas)):
            max_future = deltas_t[i:].max(dim=0).values  # (B,)
            y = torch.sigmoid(lam * (epsilon - max_future) / epsilon)
            soft_labels.append(y)

        # Compute DCM loss (monitor has gradients)
        total_bce = torch.tensor(0.0, device=device)
        n_steps = 0
        delta_prev_val = None

        for i in range(len(deltas)):
            a0_cur_d = a0_list[i + 1].detach()
            a0_prev_d = a0_list[i].detach()
            eps_d = eps_list[i + 1].detach()

            steps_done = i + 2
            progress = float(steps_done) / K
            t_idx = tau[K - 1 - (i + 1)]
            snr = float(self.alphas_cumprod[t_idx])

            features = build_dcm_features(
                a0_cur_d, a0_prev_d, eps_d, obs_cond, progress, snr
            )

            if delta_prev_val is not None:
                ratio = (deltas[i] / (delta_prev_val + 1e-8)).clamp(0, 10)
                features[:, 6] = ratio
            delta_prev_val = deltas[i]

            h = self.convergence_monitor(features, obs_cond).squeeze(1)
            total_bce = total_bce + F.binary_cross_entropy(h, soft_labels[i])
            n_steps += 1

        return total_bce / max(n_steps, 1)

    def _ddim_sample(self, obs_cond, K):
        """DDIM deterministic sampling with fixed K steps (for ablation)."""
        B = obs_cond.shape[0]
        device = obs_cond.device
        T = self.num_diffusion_iters

        tau = torch.linspace(0, T - 1, K, dtype=torch.long, device=device)
        action = torch.randn(B, self.horizon, self.action_dim, device=device)

        for i in reversed(range(len(tau))):
            t_cur = tau[i]
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps_pred = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

            alpha_bar_cur = self.alphas_cumprod[t_cur]
            a0_pred = (
                (action - torch.sqrt(1.0 - alpha_bar_cur) * eps_pred)
                / torch.sqrt(alpha_bar_cur)
            )

            if i > 0:
                alpha_bar_prev = self.alphas_cumprod[tau[i - 1]]
                action = (
                    torch.sqrt(alpha_bar_prev) * a0_pred
                    + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
                )
            else:
                action = a0_pred

        return action

    def _ddim_sample_with_dcm(self, obs_cond):
        """
        DDIM sampling with online convergence monitoring.

        At each step, computes the Tweedie denoised estimate a0_hat and feeds
        trajectory features to the ConvergenceMonitor. When the monitor's
        output exceeds the threshold, denoising halts and a0_hat is returned.
        """
        B = obs_cond.shape[0]
        device = obs_cond.device
        tau = self.ddim_tau
        K = len(tau)

        action = torch.randn(B, self.horizon, self.action_dim, device=device)
        a0_prev = None
        delta_prev = None
        steps_used = K

        for step_idx in reversed(range(K)):
            t_cur = tau[step_idx]
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
            eps_pred = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

            alpha_bar_cur = self.alphas_cumprod[t_cur]
            a0_cur = (
                (action - torch.sqrt(1.0 - alpha_bar_cur) * eps_pred)
                / torch.sqrt(alpha_bar_cur)
            )

            steps_done = K - step_idx
            if a0_prev is not None and steps_done >= self.dcm_min_steps:
                progress = float(steps_done) / K
                snr = float(alpha_bar_cur)

                features = build_dcm_features(
                    a0_cur, a0_prev, eps_pred, obs_cond, progress, snr
                )

                # Update delta_ratio feature
                diff = (a0_cur - a0_prev).reshape(B, -1)
                delta_cur = (diff ** 2).mean(dim=1, keepdim=True)
                if delta_prev is not None:
                    ratio = (delta_cur / (delta_prev + 1e-8)).clamp(0, 10)
                    features[:, 6] = ratio.squeeze(1)
                delta_prev = delta_cur

                h = self.convergence_monitor(features, obs_cond)
                if h.min().item() > self.dcm_threshold:
                    steps_used = steps_done
                    return a0_cur, steps_used
            elif a0_prev is not None:
                diff = (a0_cur - a0_prev).reshape(B, -1)
                delta_prev = (diff ** 2).mean(dim=1, keepdim=True)

            a0_prev = a0_cur

            if step_idx > 0:
                alpha_bar_prev = self.alphas_cumprod[tau[step_idx - 1]]
                action = (
                    torch.sqrt(alpha_bar_prev) * a0_cur
                    + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
                )
            else:
                action = a0_cur

        return action, steps_used

    def _ddpm_sample(self, obs_cond):
        """Original DDPM reverse sampling (num_diffusion_iters steps)."""
        B = obs_cond.shape[0]
        action = torch.randn(B, self.horizon, self.action_dim, device=obs_cond.device)

        for t in reversed(range(self.num_diffusion_iters)):
            t_batch = torch.full((B,), t, device=obs_cond.device, dtype=torch.long)
            noise_pred = self.noise_pred_net(action, t_batch, global_cond=obs_cond)

            pred_x0 = (
                self.sqrt_recip_alphas_cumprod[t] * action -
                self.sqrt_recipm1_alphas_cumprod[t] * noise_pred
            )

            if t > 0:
                posterior_mean = (
                    self.posterior_mean_coef1[t] * pred_x0 +
                    self.posterior_mean_coef2[t] * action
                )
                action = posterior_mean + torch.exp(
                    0.5 * self.posterior_log_variance_clipped[t]
                ) * torch.randn_like(action)
            else:
                action = pred_x0

        return action

    def get_action(self, data):
        """
        Inference: generate action.

        If DCM enabled: DDIM with convergence-based early stopping.
        Otherwise: standard DDPM with full num_diffusion_iters steps.
        """
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            obs_cond = self.encode_obs(data)

            if self.dcm_enabled:
                action, _ = self._ddim_sample_with_dcm(obs_cond)
            else:
                action = self._ddpm_sample(obs_cond)

        return action[:, 0, :].detach().cpu().numpy()

    def reset(self):
        self.obs_history = {}
