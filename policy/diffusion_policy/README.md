# Diffusion Policy 集成说明（基于 LIBERO 框架）

本目录包含集成到 LIBERO 基准测试框架中的 Diffusion Policy 实现。

## 目录结构

```
LIBERO/
├── policy/
│   └── diffusion_policy/
│       ├── __init__.py
│       └── diffusion_policy.py       # Diffusion Policy 模型实现
├── libero/
│   ├── configs/
│   │   └── policy/
│   │       └── diffusion_policy.yaml # Diffusion Policy 配置文件
│   └── lifelong/
│       └── algos/
│           └── base.py               # 已修改：支持 eval.eval=false，修复空 losses 崩溃
├── train.py                          # 训练入口脚本（已修改：支持 eval.eval=false）
└── eval_vis.py                       # 带 MuJoCo 实时窗口的可视化评估脚本
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

# Diffusion Policy baseline
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=10 +num_gpus=4

# 其他 baseline 同理
python train.py policy=bc_transformer_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false'

# 4 数据集 × 多策略 × 3 seeds，建议多卡并行
```

### 快速调试（Sequential）

```bash
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL \
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
| `train.n_epochs=50` | 训练轮数（默认 50，Diffusion Policy 需要 500-1000） |
| `task_embedding_format=one-hot` | 跳过 BERT，网络受限时使用 |

---

## 权重保存位置

### Multitask（产出 1 个模型）

```
experiments/
└── LIBERO_SPATIAL/
    └── Multitask/
        └── DiffusionPolicy_seed0/
            └── run_001/
                └── multitask_model.pth   # 唯一权重，在所有 10 个任务上评估
```

### Sequential（产出 10 个模型）

```
experiments/
└── LIBERO_SPATIAL/
    └── Sequential/
        └── DiffusionPolicy_seed42/
            └── run_001/
                ├── task0_model.pth   # Task 1（仅在 Task 1 上表现好）
                ├── ...
                └── task9_model.pth   # Task 10（可能遗忘前 9 个任务）
```

- `eval.eval=true`：保存每任务**成功率最高** epoch 的权重
- `eval.eval=false`：保存每任务**最后一个** epoch 的权重

---

## 评估

### 可视化评估（带 MuJoCo 实时窗口）

```bash
# 需要显示器或 X11 转发（ssh -X）
export DISPLAY=:0

# Multitask 模型
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Multitask/DiffusionPolicy_seed0/run_001/multitask_model.pth \
  --task_id 0 \
  --n_eval 5

# Sequential 模型
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Sequential/DiffusionPolicy_seed42/run_001/task9_model.pth \
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

| 策略名 | 类型 |
|--------|------|
| `bc_rnn_policy` | 基于 LSTM 的行为克隆 |
| `bc_transformer_policy` | 基于 Transformer 的行为克隆 |
| `bc_vilt_policy` | 基于 ViLT 的行为克隆 |
| `diffusion_policy` | Diffusion Policy（新增） |

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

**语义**：用最近 2 帧观测 (obs₀, obs₁) 预测接下来 8 步动作 (act₂ ~ act₉)

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
│ forward: 预测 actions[2:10] (共 horizon=8 步)                │
│                                                              │
│   actions[2:10] (8, 7)                                       │
│        │                                                     │
│        ▼ 加噪 (timestep t)                                   │
│   noisy_actions (8, 7)                                       │
│        │                                                     │
│        ▼ ConditionalUNet1D(noisy, t, obs_cond)              │
│   noise_pred (8, 7)                                          │
│        │                                                     │
│        ▼ MSE(noise_pred, noise)                              │
│   loss                                                       │
└─────────────────────────────────────────────────────────────┘

推理时:
   obs_history = [obs_{t-1}, obs_t]  (维护 2 帧滚动窗口)
        │
        ▼ encode_obs
   obs_cond (256)
        │
        ▼ UNet 100 步去噪
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
| **UNet 1D** | 动作序列去噪 | (B, horizon, action_dim) |

### 与 BCRNNPolicy 的关键区别

| 组件 | BCRNNPolicy | DiffusionPolicy |
|------|-------------|-----------------|
| LSTM 输出 | 每帧独立预测动作 | 作为 UNet 的全局条件 |
| 预测目标 | `actions[i]` | `actions[n_obs_steps:n_obs_steps+horizon]` |
| 推理执行 | 单步 | 从 8 步序列中取第 1 步 |

---

## 已知修改

| 文件 | 修改内容 |
|------|---------|
| `libero/lifelong/algos/base.py` | 支持 `eval.eval=false`；修复空 losses IndexError |
| `train.py` | Multitask 算法条件分支 |
| `policy/diffusion_policy/diffusion_policy.py` | **修复观测-动作时序对齐问题**（详见下文） |

---

## 关键 Bug 修复：观测-动作时序对齐

### 问题分析

原实现存在**严重的时序错位问题**：

```python
# 错误实现
def forward(self, data):
    obs_cond = self.encode_obs(data)        # LSTM 处理 obs[0:10]，编码 obs[9]
    action = data["actions"][:, :8, :]      # 目标是 actions[0:8]（obs[0] 的动作）
    # obs_cond 编码 obs[9]，但目标是 obs[0] 的动作 → 完全错位！
```

**后果**：模型学习的是"看到当前状态，预测 9 步前的动作"，完全无法收敛。

### 正确实现

```python
def encode_obs(self, data):
    # 训练时只用前 n_obs_steps 帧
    x = data["obs"][img_name]
    if T > 1:
        x = x[:, :self.n_obs_steps, ...]  # obs[0:2]
    # 推理时维护 n_obs_steps 的滚动窗口
    ...
    return h_n[-1]  # 编码 obs[0], obs[1]

def forward(self, data):
    obs_cond = self.encode_obs(data)                        # 编码 obs[0:2]
    action = data["actions"][:, n_obs_steps:n_obs_steps+horizon, :]  # actions[2:10]
    # 正确：用 obs[0:2] 预测 actions[2:10]
```

### 数据流对比

| 阶段 | 错误实现 | 正确实现 |
|------|---------|---------|
| 观测条件 | obs[0:10] → LSTM → 编码 obs[9] | obs[0:2] → LSTM → 编码 obs[0,1] |
| 动作目标 | actions[0:8] | actions[2:10] |
| 语义 | 用当前状态预测过去的动作 | 用最近 2 帧预测未来 8 步 |

### 配置约束

```python
# 必须满足
assert seq_len >= n_obs_steps + horizon
# 当前配置: 10 >= 2 + 8 ✓
```

---

## 训练命令工作流分析

以下命令的完整执行流程分析：

```bash
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=1000
```

### 阶段 1：配置加载（Hydra）

```
train.py (main)
    │
    ├── @hydra.main(config_path="libero/configs", config_name="config")
    │   加载 libero/configs/config.yaml
    │
    ├── CLI 参数覆盖：
    │   ├── policy=diffusion_policy          → 加载 configs/policy/diffusion_policy.yaml
    │   ├── benchmark_name=LIBERO_SPATIAL    → 设置 benchmark_name
    │   ├── lifelong=multitask               → 加载 configs/lifelong/multitask.yaml
    │   ├── seed=0                           → 全局随机种子
    │   ├── eval.eval=false                  → 禁用训练中评估
    │   └── train.n_epochs=1000              → 训练 1000 轮
    │
    └── 转换为 EasyDict cfg
```

**最终合并的配置**：
```yaml
seed: 0
benchmark_name: LIBERO_SPATIAL
device: cuda
data:
  seq_len: 10
policy:
  policy_type: DiffusionPolicy
  num_diffusion_iters: 100
  horizon: 8
  n_obs_steps: 2
  image_embed_size: 64
  text_embed_size: 64
  obs_hidden_size: 256
  down_dims: [128, 256, 512]
lifelong:
  algo: Multitask
  eval_in_train: false
eval:
  eval: false
train:
  n_epochs: 1000
  batch_size: 32
```

---

### 阶段 2：数据集加载

```
train.py (main)
    │
    ├── benchmark = get_benchmark("LIBERO_SPATIAL")(task_order_index)
    │   └── n_tasks = 10
    │
    └── for i in range(10):
        └── get_dataset(dataset_path, obs_modality, seq_len=10)
            └── SequenceDataset 返回:
                ├── obs["agentview_rgb"]: (10, 3, 128, 128)
                ├── obs["joint_states"]: (10, 7)
                ├── obs["gripper_states"]: (10, 2)
                └── actions: (10, 7)
```

**关键**：数据集返回 `seq_len=10` 的序列，模型内部截取需要的部分。

---

### 阶段 3：训练循环（Multitask）

```python
for epoch in range(1, 1001):
    for data in train_dataloader:
        # data["obs"][img]: (B, 10, 3, 128, 128)
        # data["actions"]: (B, 10, 7)
        
        loss = policy.compute_loss(data)
        # 内部:
        #   obs_cond = encode_obs(data)  # 只用 obs[:2]
        #   action_target = data["actions"][:, 2:10, :]  # actions[2:10]
        #   loss = diffusion_loss(action_target, obs_cond)
        
        loss.backward()
        optimizer.step()
```

---

### 时间与资源估算

**LIBERO_SPATIAL（10 任务）**：

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据集加载 | ~2 分钟 | 10 个 HDF5 文件 |
| Epoch 1-1000 | ~8 小时 | 单卡 A100 |
| **总计** | **~8.5 小时** | |

**GPU 显存**：~8GB（batch_size=32）

---

## 参考文献

- **Diffusion Policy**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- **LIBERO 基准**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
