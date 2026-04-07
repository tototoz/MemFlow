# ACT Policy 集成说明（基于 LIBERO 框架）

本目录包含集成到 LIBERO 基准测试框架中的 ACT (Action Chunking Transformer) Policy 实现。

## 目录结构

```
LIBERO/
├── policy/
│   └── act/
│       ├── __init__.py
│       └── act_policy.py           # ACT Policy 模型实现
├── libero/
│   └── configs/
│       └── policy/
│           └── act_policy.yaml     # ACT Policy 配置文件
└── train.py                        # 训练入口脚本
```

---

## ACT vs Diffusion vs Flow Matching 对比

| 特性 | BC (单步) | ACT | Diffusion | Flow Matching |
|------|----------|-----|-----------|---------------|
| **预测方式** | 单步动作 | 动作序列 | 去噪过程 | ODE 积分 |
| **预测长度** | 1 | 8 (可调) | 8 (可调) | 8 (可调) |
| **推理步数** | 1 | 1 | 100+ | 10-20 |
| **模型架构** | MLP/RNN | CVAE+Transformer | UNet | UNet |
| **动作平滑** | 无 | 时序聚合 | 天然平滑 | 天然平滑 |
| **推理速度** | 最快 | 快 | 慢 | 较快 |

**核心优势**：ACT 通过 CVAE 学习动作分布，结合 Transformer 编码器-解码器，可以一次性预测完整的动作序列，并支持时序聚合实现平滑执行。

---

## 算法原理

### 训练：CVAE (Conditional Variational Autoencoder)

ACT 使用 CVAE 学习条件动作分布：

```
编码器 q(z|a):
  actions (B, horizon, action_dim) → Transformer Encoder → mu, logvar
  z ~ N(mu, diag(exp(logvar)))

解码器 p(a|z, obs):
  z (B, latent_dim) + obs_cond (B, cond_dim)
  → Transformer Decoder → actions (B, horizon, action_dim)

损失函数:
  L = L_recon + kl_weight * KL(q(z|a) || N(0, I))
  其中 L_recon 通常使用 L1 loss
```

### 推理：采样 + 时序聚合

```python
# 1. 从先验采样 z ~ N(0, I)
z = torch.randn(B, latent_dim)

# 2. 解码得到动作序列
actions = decoder(z, obs_cond)  # (B, horizon, action_dim)

# 3. 时序聚合（可选）
# 使用指数加权平均平滑相邻 chunk 的重叠动作
```

---

## 模型架构

### 完整架构图

```
训练数据: obs[0:10]  actions[0:10]
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ encode_obs: 只用前 n_obs_steps=2 帧                          │
│                                                              │
│   obs[0:2]                                                   │
│     │                                                        │
│     ├── agentview_rgb (2,3,128,128) → ResNet+FiLM → (2,64)  │
│     ├── eye_in_hand_rgb (2,3,128,128) → ResNet+FiLM → (2,64)│
│     ├── joint+gripper (2,9) ────────────────────→ (2,9)     │
│     └── task_emb (768) → MLP → (64) → expand → (2,64)       │
│              │                                               │
│              ▼ concat                                        │
│         (2, 201)                                             │
│              │                                               │
│              ▼ LSTM                                          │
│         obs_cond (256)  ← 编码 obs[0], obs[1] 的信息         │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│ forward: CVAE 训练                                           │
│                                                              │
│   actions[1:9] (8, 7)  ← action target (horizon=8)          │
│        │                                                     │
│        ▼ CVAE Encoder (Transformer)                          │
│   mu (32), logvar (32)                                       │
│        │                                                     │
│        ▼ 重参数化 z = mu + eps * exp(0.5 * logvar)           │
│   z (32)                                                     │
│        │                                                     │
│        ▼ CVAE Decoder (Transformer, cond=obs_cond)          │
│   actions_pred (8, 7)                                        │
│        │                                                     │
│        ▼ L1(actions_pred, actions) + kl_weight * KL          │
│   loss                                                       │
└─────────────────────────────────────────────────────────────┘

推理时:
   obs_history = [obs_{t-1}, obs_t]  (维护 2 帧滚动窗口)
        │
        ▼ encode_obs
   obs_cond (256)
        │
        ▼ z ~ N(0, I), Decoder
   actions[0:8] = [act_{t+1}, ..., act_{t+8}]
        │
        ▼ 时序聚合 / 直接取第一个
   执行 act_{t+1} (下一步动作)
```

### 关键组件说明

| 组件 | 作用 | 参数 |
|------|------|------|
| **CVAE Encoder** | 将动作序列编码为潜在变量 z | 4层 Transformer, hidden=256 |
| **CVAE Decoder** | 从 z 和 obs_cond 解码动作序列 | 7层 Transformer, hidden=256 |
| **Latent dim** | 潜在变量维度 | 32 |
| **KL weight** | KL 散度权重 | 10.0 |

---

## 算法选择说明

LIBERO 框架支持多种终身学习算法（`lifelong=xxx`）：

| 算法 | 数据策略 | 产出模型数 | 适用场景 |
|------|---------|----------|---------|
| `Sequential`（默认） | 逐任务覆盖 | 10 个 | 终身学习 baseline |
| **`Multitask`（推荐）** | **所有任务混合** | **1 个** | **策略能力对比** |
| `ER` | 回放样本 | 10 个 | 抗遗忘研究 |
| `EWC` | 参数正则 | 10 个 | 抗遗忘研究 |

---

## 训练

### 论文实验（推荐：Multitask）

```bash
export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0

# ACT Policy
python train.py policy=act_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=300

# Diffusion Policy（对比）
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=1000

# Flow Matching Policy（对比）
python train.py policy=flow_matching_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500
```

> **注意**：ACT 通常收敛较快，300 epochs 即可达到较好效果。

### 快速调试（Sequential）

```bash
python train.py policy=act_policy benchmark_name=LIBERO_SPATIAL \
  seed=42 'eval.eval=false'
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `lifelong=multitask` | 混合所有任务训练（推荐） |
| `eval.eval=false` | 跳过训练中评估 |
| `train.n_epochs=300` | 训练轮数（ACT 建议 300） |
| `kl_weight=10.0` | KL 散度权重 |

---

## 权重保存位置

### Multitask（产出 1 个模型）

```
experiments/
└── LIBERO_SPATIAL/
    └── Multitask/
        └── ACTPolicy_seed0/
            └── run_001/
                └── multitask_model.pth
```

---

## 评估

```bash
export DISPLAY=:0

# ACT Policy
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Multitask/ACTPolicy_seed0/run_001/multitask_model.pth \
  --task_id 0 \
  --n_eval 5
```

---

## 配置参数说明

```yaml
# libero/configs/policy/act_policy.yaml

policy_type: ACTPolicy

# ACT 特有参数
horizon: 8                   # 动作序列长度
n_obs_steps: 2               # 观测条件帧数
latent_dim: 32               # CVAE 潜在变量维度
kl_weight: 10.0              # KL 散度权重
temporal_agg: true           # 时序聚合
ema_alpha: 0.5               # EMA 衰减系数

# 观测编码
image_embed_size: 64
text_embed_size: 64
obs_hidden_size: 256

# Transformer 架构
hidden_dim: 256
nhead: 8
encoder_n_layers: 4          # CVAE 编码器层数
decoder_n_layers: 7          # CVAE 解码器层数
```

---

## 时序聚合 (Temporal Aggregation)

ACT 支持时序聚合来平滑动作执行：

```
时间步 t: 预测 [a_t, a_{t+1}, ..., a_{t+7}]
时间步 t+1: 预测 [a_{t+1}', a_{t+2}', ..., a_{t+8}']

时序聚合:
  a_{t+1}^{final} = α * a_{t+1} + (1-α) * a_{t+1}'
```

启用方式：设置 `temporal_agg: true`，调整 `ema_alpha` 控制权重。

---

## 可用策略对比

| 策略名 | 类型 | 推理步数 | 预测长度 |
|--------|------|---------|---------|
| `bc_rnn_policy` | LSTM 行为克隆 | 1 | 1 |
| `bc_transformer_policy` | Transformer 行为克隆 | 1 | 1 |
| `diffusion_policy` | Diffusion Policy | 100+ | 8 |
| `flow_matching_policy` | Flow Matching | 10-20 | 8 |
| **`act_policy`** | **ACT (CVAE+Transformer)** | **1** | **8** |

---

## 时间与资源估算

**LIBERO_SPATIAL（10 任务）**：

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据集加载 | ~2 分钟 | 10 个 HDF5 文件 |
| Epoch 1-300 | ~2.5 小时 | 单卡 A100（比 Diffusion 快 3 倍） |
| **总计** | **~3 小时** | |

**GPU 显存**：~5GB（batch_size=32）

**推理速度**：
- BC Policy: ~0.01s/action
- ACT Policy: ~0.02s/action（解码整个 chunk）
- Flow Matching: ~0.05s/action
- Diffusion Policy: ~0.5s/action

---

## 参考文献

- **ACT**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) (Zhao et al., RSS 2023)
- **Diffusion Policy**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **LIBERO 基准**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
