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
    1D U-Net for diffusion policy (official architecture).

    Args:
        input_dim: action dimension
        global_cond_dim: observation feature dimension
        diffusion_step_embed_dim: timestep embedding size
        down_dims: channel sizes at each U-Net level, e.g. [128, 256, 512]
        kernel_size: conv kernel size
        n_groups: GroupNorm groups
        cond_predict_scale: FiLM scale+bias (True) or bias only (False)
    """
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

        # Up blocks (reversed in_out[1:])
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
        """
        sample:      (B, T, input_dim)  noisy action
        timestep:    (B,) or scalar
        global_cond: (B, global_cond_dim)
        returns:     (B, T, input_dim)
        """
        # (B, T, D) → (B, D, T) for Conv1d
        x = einops.rearrange(sample, 'b h t -> b t h')

        # Time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(x.device)
        timestep = timestep.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # Down path — save skip connections BEFORE downsample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)           # save after resnet2, before downsample
            x = downsample(x)

        # Mid
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Up path — cat with skip, then conv, then upsample
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        # (B, D, T) → (B, T, D)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


# ============================================================
# DiffusionPolicy
# ============================================================

class DiffusionPolicy(BasePolicy):
    """
    Diffusion Policy for robot manipulation.
    Uses DDPM with a conditional 1D U-Net for denoising.
    """
    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        # Diffusion hyperparameters
        self.num_diffusion_iters = policy_cfg.num_diffusion_iters
        self.horizon = policy_cfg.horizon
        self.n_obs_steps = policy_cfg.n_obs_steps

        # Verify alignment: seq_len must be >= n_obs_steps + horizon - 1
        # We use obs[:n_obs_steps] to predict actions[n_obs_steps-1 : n_obs_steps-1+horizon]
        required_len = self.n_obs_steps + self.horizon - 1
        assert cfg.data.seq_len >= required_len, (
            f"seq_len ({cfg.data.seq_len}) must be >= n_obs_steps + horizon - 1 "
            f"({self.n_obs_steps} + {self.horizon} - 1 = {required_len})"
        )

        # History buffer for inference: rolling window of n_obs_steps frames
        self.obs_history = {}  # {obs_name: deque(maxlen=n_obs_steps)}

        # Linear noise schedule
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

        self.image_encoders = self.obs_encoders  # compatibility with BasePolicy

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

        # LSTM to encode observation sequence → fixed-size vector
        self.obs_temporal_encoder = nn.LSTM(
            input_size=obs_dim,
            hidden_size=policy_cfg.obs_hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.final_obs_dim = policy_cfg.obs_hidden_size

        # Action dimension
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

        # Move all submodules and buffers to target device
        self.to(self.device)

    def encode_obs(self, data):
        """Encode observations to a condition vector (B, obs_hidden_size).

        Always uses only the first n_obs_steps frames as condition.
        - Training: data["obs"][img] is (B, seq_len, ...), take [:n_obs_steps]
        - Inference: data["obs"][img] is (B, 1, ...), maintain rolling window of n_obs_steps
        """
        encoded = []

        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape

            if T == 1:
                # Inference: maintain rolling window of n_obs_steps frames
                from collections import deque
                if img_name not in self.obs_history:
                    self.obs_history[img_name] = deque(maxlen=self.n_obs_steps)
                self.obs_history[img_name].append(x.clone())

                history_list = list(self.obs_history[img_name])
                if len(history_list) < self.n_obs_steps:
                    pad_len = self.n_obs_steps - len(history_list)
                    history_list = [history_list[0]] * pad_len + history_list
                x = torch.cat(history_list, dim=1)  # (B, n_obs_steps, C, H, W)
                T = x.shape[1]
            else:
                # Training: use only first n_obs_steps frames
                x = x[:, :self.n_obs_steps, :, :, :]
                T = self.n_obs_steps

            e = self.image_encoders[img_name](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"].reshape(B, 1, -1).repeat(1, T, 1).reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

        # Extra modalities (joint_states, gripper_states)
        extra = self.extra_encoder(data["obs"])  # (B, T_extra, extra_dim)
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
            extra = torch.cat(history_list, dim=1)  # (B, n_obs_steps, extra_dim)
        else:
            extra = extra[:, :self.n_obs_steps, :]  # training: truncate to n_obs_steps

        encoded.append(extra)

        lang_h = self.language_encoder(data)             # (B, text_dim)
        T_obs = encoded[0].shape[1]
        encoded.append(lang_h.unsqueeze(1).expand(-1, T_obs, -1))

        x = torch.cat(encoded, dim=-1)                  # (B, n_obs_steps, obs_dim)
        _, (h_n, _) = self.obs_temporal_encoder(x)
        return h_n[-1]                                   # (B, obs_hidden_size)

    def forward(self, data, train_mode=True):
        """Training forward: predict noise.

        Design: use the first n_obs_steps frames as observation condition,
        predict the next horizon frames of actions.

        Crucial alignment:
        - obs_cond encodes obs[0:n_obs_steps], with the LAST frame being obs[n_obs_steps-1]
        - We should predict actions starting from obs[n_obs_steps-1]'s action
        - In robomimic: actions[i] is the action taken AT obs[i]
        - So the prediction target is actions[n_obs_steps-1 : n_obs_steps-1+horizon]

        data["obs"] shape: (B, seq_len, ...)  where seq_len >= n_obs_steps + horizon - 1
        - obs_cond comes from obs[:n_obs_steps]
        - action target is actions[n_obs_steps-1 : n_obs_steps-1+horizon]
        """
        data = self.preprocess_input(data, train_mode=True)
        obs_cond = self.encode_obs(data)        # (B, obs_hidden_size)

        # Action target: starting from the LAST observation frame's action
        start_idx = self.n_obs_steps - 1  # = 1 when n_obs_steps=2
        action = data["actions"][:, start_idx : start_idx + self.horizon, :]  # (B, horizon, action_dim)
        B = action.shape[0]

        t = torch.randint(0, self.num_diffusion_iters, (B,), device=action.device)
        noise = torch.randn_like(action)

        noisy_action = (
            self.sqrt_alphas_cumprod[t, None, None] * action +
            self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        )

        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs_cond)
        return {"noise": noise, "noise_pred": noise_pred}

    def compute_loss(self, data, reduction="mean"):
        output = self.forward(data, train_mode=True)
        return F.mse_loss(output["noise_pred"], output["noise"], reduction=reduction)

    def get_action(self, data):
        """Inference: iterative denoising from Gaussian noise."""
        self.eval()
        data = self.preprocess_input(data, train_mode=False)

        with torch.no_grad():
            # encode_obs will handle history buffer internally
            obs_cond = self.encode_obs(data)
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

        return action[:, 0, :].detach().cpu().numpy()

    def reset(self):
        """Clear observation history buffer."""
        self.obs_history = {}
