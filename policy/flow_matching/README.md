# Flow Matching Policy 集成说明（基于 LIBERO 框架）

本目录包含集成到 LIBERO 基准测试框架中的 Flow Matching Policy 实现。

## 目录结构

```
LIBERO/
├── policy/
│   └── flow_matching/
│       ├── __init__.py
│       └── flow_matching_policy.py  # Flow Matching Policy 模型实现
├── libero/
│   └── configs/
│       └── policy/
│           └── flow_matching_policy.yaml  # Flow Matching Policy 配置文件
└── train.py                          # 训练入口脚本
```

---

## Flow Matching vs Diffusion 对比

| 特性 | Diffusion Policy | Flow Matching Policy |
|------|------------------|----------------------|
| **训练目标** | 预测噪声 ε | 预测向量场 v |
| **推理方式** | 迭代去噪 (DDPM/DDIM) | ODE 积分 (Euler) |
| **推理步数** | 100+ | 10-20 |
| **数学基础** | SDE / Score Matching | ODE / Optimal Transport |
| **路径** | 随机噪声路径 | 确定性最优传输路径 |
| **推理速度** | 较慢 | **更快** |

**核心优势**：Flow Matching 使用最优传输路径，可以用更少的积分步数达到与 Diffusion 相当的效果。

---

## 算法原理

### 训练：Conditional Flow Matching (CFM)

Flow Matching 学习一个向量场 `v_t(x_t, t)`，将噪声传输到数据分布：

```
最优传输路径：
  x_t = (1 - t) * x_0 + t * x_1

其中：
  - x_0 ~ N(0, I) 是噪声
  - x_1 = action 是真实数据
  - t ∈ [0, 1] 是时间

目标向量场：
  v_t = x_1 - x_0 = action - noise

训练损失：
  L = E[||v_θ(x_t, t) - v_t||²]
```

### 推理：Euler 积分

```python
x = torch.randn(...)  # 从噪声开始
dt = 1.0 / num_integration_steps
for i in range(num_integration_steps):
    t = i / num_integration_steps
    v = vector_field_net(x, t, obs_cond)
    x = x + v * dt  # Euler step
return x[:, 0, :]    # 返回第一个动作
```

---

## 算法选择说明

LIBERO 框架支持多种终身学习算法（`lifelong=xxx`），选哪个取决于实验目的：

| 算法 | 数据策略 | 产出模型数 | 适用场景 |
|------|---------|----------|---------|
| `Sequential`（默认） | 逐任务覆盖，只用当前任务数据 | 10 个（每任务一个） | 终身学习 baseline，遗忘严重 |
| **`Multitask`（推荐）** | **所有任务数据混合训练** | **1 个** | **策略能力对比，无遗忘干扰** |
| `ER` | 当前任务 + 旧任务回放样本 | 10 个 | 抗遗忘研究 |
| `EWC` | 当前任务数据 + 参数正则 | 10 个 | 抗遗忘研究 |

> **论文实验推荐用 `Multitask`**：MemFlow 的记忆解决的是单个 episode 内的任务进度追踪，与跨任务遗忘无关。用 Multitask 可排除遗忘干扰，干净地对比各策略能力。

---

## 训练

### 论文实验（推荐：Multitask）

```bash
export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0

# Flow Matching Policy baseline
python train.py policy=flow_matching_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500

# Diffusion Policy baseline（对比）
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=1000

# 其他 baseline 同理
python train.py policy=bc_transformer_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false'
```

> **注意**：Flow Matching 通常收敛更快，500 epochs 即可达到较好效果。

### 快速调试（Sequential）

```bash
python train.py policy=flow_matching_policy benchmark_name=LIBERO_SPATIAL \
  seed=42 'eval.eval=false'
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `lifelong=multitask` | 混合所有任务训练（论文实验推荐） |
| `lifelong=sequential` | 逐任务顺序训练（默认） |
| `eval.eval=false` | 跳过训练中评估，速度快（推荐） |
| `eval.use_mp=false` | 单进程评估，避免 EGL 崩溃 |
| `eval.n_eval=5` | 每次评估只跑 5 个 rollout |
| `train.n_epochs=500` | 训练轮数（Flow Matching 建议 500） |
| `task_embedding_format=one-hot` | 跳过 BERT，网络受限时使用 |

---

## 权重保存位置

### Multitask（产出 1 个模型）

```
experiments/
└── LIBERO_SPATIAL/
    └── Multitask/
        └── FlowMatchingPolicy_seed0/
            └── run_001/
                └── multitask_model.pth   # 唯一权重，在所有 10 个任务上评估
```

### Sequential（产出 10 个模型）

```
experiments/
└── LIBERO_SPATIAL/
    └── Sequential/
        └── FlowMatchingPolicy_seed42/
            └── run_001/
                ├── task0_model.pth   # Task 1
                ├── ...
                └── task9_model.pth   # Task 10
```

---

## 评估

### 可视化评估（带 MuJoCo 实时窗口）

```bash
# 需要显示器或 X11 转发（ssh -X）
export DISPLAY=:0

# Multitask 模型
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Multitask/FlowMatchingPolicy_seed0/run_001/multitask_model.pth \
  --task_id 0 \
  --n_eval 5

# Sequential 模型
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Sequential/FlowMatchingPolicy_seed42/run_001/task9_model.pth \
  --task_id 0 \
  --n_eval 5
```

| 参数 | 说明 |
|------|------|
| `--model_path` | 权重文件路径 |
| `--task_id` | 评估哪个任务（0-9） |
| `--n_eval` | episode 数，默认 5 |
| `--device` | GPU 设备，默认 `cuda:0` |

> 无显示器服务器：
> ```bash
> sudo apt install xvfb && Xvfb :0 -screen 0 1280x1024x24 &
> export DISPLAY=:0
> ```

---

## 可用基准

| 基准名称 | 任务数 | 说明 |
|---------|--------|------|
| `LIBERO_SPATIAL` | 10 | 空间推理（同物体不同位置） |
| `LIBERO_OBJECT` | 10 | 物体操作（同场景不同目标） |
| `LIBERO_GOAL` | 10 | 目标条件（多种操作目标） |
| `LIBERO_100` | 100 | 大规模综合基准 |

## 可用策略

| 策略名 | 类型 | 推理步数 |
|--------|------|---------|
| `bc_rnn_policy` | 基于 LSTM 的行为克隆 | 1 |
| `bc_transformer_policy` | 基于 Transformer 的行为克隆 | 1 |
| `bc_vilt_policy` | 基于 ViLT 的行为克隆 | 1 |
| `diffusion_policy` | Diffusion Policy | 100+ |
| **`flow_matching_policy`** | **Flow Matching Policy** | **10-20** |

---

## 模型架构

### 核心设计：观测-动作时序对齐

**关键参数**：
- `seq_len = 10`：数据集返回的序列长度
- `n_obs_steps = 2`：作为条件的观测帧数
- `horizon = 8`：预测的动作序列长度

**时序对齐要求**：`seq_len >= n_obs_steps + horizon`

```
时间轴:    t-1    t    t+1   t+2   ...  t+7   t+8   t+9
           │      │     │     │          │     │     │
观测:     obs₀  obs₁                             (余下不用)
           │─────│
           └─ 作为条件 (n_obs_steps=2)

动作:           act₂  act₃  act₄  ...  act₉
                 │─────────────────────│
                 └── 预测目标 (horizon=8)
```

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
│ forward: Flow Matching 训练                                  │
│                                                              │
│   actions[2:10] (8, 7)  ← x_1 (真实数据)                     │
│        │                                                     │
│        ▼ 采样 t ~ Uniform(0, 1)                              │
│        ▼ 采样 x_0 ~ N(0, I)                                  │
│        ▼ 插值 x_t = (1-t)*x_0 + t*x_1                        │
│   x_t (8, 7)                                                 │
│        │                                                     │
│        ▼ ConditionalUnet1D(x_t, t, obs_cond)                 │
│   v_pred (8, 7)  ← 预测向量场                                │
│        │                                                     │
│        ▼ MSE(v_pred, x_1 - x_0)                              │
│   loss                                                       │
└─────────────────────────────────────────────────────────────┘

推理时:
   obs_history = [obs_{t-1}, obs_t]  (维护 2 帧滚动窗口)
        │
        ▼ encode_obs
   obs_cond (256)
        │
        ▼ Euler 积分 10 步
   actions[0:8] = [act_{t+1}, ..., act_{t+8}]
        │
        ▼ 取 actions[0] 执行
   执行 act_{t+1} (下一步动作)
```

### 关键组件说明

| 组件 | 作用 | 输出维度 |
|------|------|---------|
| **ResNet+FiLM** | 图像编码，语言条件注入 | (B, n_obs_steps, 64) |
| **MLP** | 语言任务嵌入降维 | (B, 64) |
| **LSTM** | 观测序列时序编码 | (B, 256) |
| **UNet 1D** | 向量场预测 | (B, horizon, action_dim) |

### 与 DiffusionPolicy 的关键区别

| 组件 | DiffusionPolicy | FlowMatchingPolicy |
|------|-----------------|-------------------|
| UNet 输出 | 噪声预测 ε | 向量场 v |
| 训练目标 | 预测添加的噪声 | 预测 x_1 - x_0 |
| 推理过程 | DDPM/DDIM 去噪 | Euler ODE 积分 |
| 推理步数 | 100+ | 10-20 |
| 路径类型 | 随机扩散路径 | 确定性最优传输 |

---

## 配置参数说明

```yaml
# libero/configs/policy/flow_matching_policy.yaml

policy_type: FlowMatchingPolicy

# Flow Matching 特有参数
num_integration_steps: 10    # Euler 积分步数（比 diffusion 少 10 倍）
horizon: 8                   # 预测动作序列长度
n_obs_steps: 2               # 观测条件帧数

# 观测编码
image_embed_size: 64         # 图像嵌入维度
text_embed_size: 64          # 文本嵌入维度
obs_hidden_size: 256         # LSTM 隐藏维度

# UNet 架构
time_embed_dim: 256          # 时间嵌入维度
down_dims: [128, 256, 512]   # U-Net 各层通道数
kernel_size: 3               # 卷积核大小
n_groups: 8                  # GroupNorm 分组数
cond_predict_scale: false    # FiLM 是否预测 scale（False 只预测 bias）
```

---

## 与 MemFlow 设计的关联

此实现是 README.md 中描述的 **MemFlow Flow Matching Policy** 的基础版本。

### 已实现

- ✅ Conditional Flow Matching 训练目标
- ✅ Optimal Transport 插值路径
- ✅ Euler ODE 积分推理
- ✅ 观测-动作时序对齐

### 待实现（MemFlow 计划）

- ❌ L1 Working Memory：因果 Transformer 编码最近 (obs, action) 对
- ❌ L2 Episodic Memory：事件驱动稀疏记忆库 + Cross-Attention 读取
- ❌ Adaptive Step Router：MLP 从 {2, 4, 8, 16} 动态选择积分步数

当前 `num_integration_steps` 是固定值，MemFlow 计划引入自适应步数路由器。

---

## 时间与资源估算

**LIBERO_SPATIAL（10 任务）**：

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据集加载 | ~2 分钟 | 10 个 HDF5 文件 |
| Epoch 1-500 | ~4 小时 | 单卡 A100（比 Diffusion 快 2 倍） |
| **总计** | **~4.5 小时** | |

**GPU 显存**：~6GB（batch_size=32，比 Diffusion 省显存）

**推理速度**：
- Diffusion Policy: ~0.5s/action（100 步去噪）
- Flow Matching: ~0.05s/action（10 步积分）

---

## 参考文献

- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., ICLR 2023)
- **Conditional Flow Matching**: [Improving and Generalizing Flow-Based Generative Models with Minimalistic Probability Training](https://arxiv.org/abs/2306.05448) (Tong et al., ICLR 2024)
- **Diffusion Policy**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- **LIBERO 基准**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
