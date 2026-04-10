# MemFlow Policy 集成说明（基于 LIBERO 框架）

本目录包含集成到 LIBERO 基准测试框架中的 **Memory-Augmented Flow Matching Policy** 实现。

MemFlow 在 Flow Matching Policy 基础上引入层级记忆架构，解决长 horizon 任务中的阶段混淆和动作抖动问题。

## 目录结构

```
LIBERO/
├── policy/
│   └── memflow/
│       ├── __init__.py
│       ├── memflow_policy.py     # MemFlow Policy 模型实现
│       └── README.md             # 本文档
├── libero/
│   └── configs/
│       └── policy/
│           └── memflow_policy.yaml  # MemFlow 配置文件
├── scripts/
│   └── download_dinov2.py        # DINOv2 预下载脚本
└── train.py                      # 训练入口脚本
```

---

## 快速开始

```bash
# 训练（开启两个记忆模块）
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500 \
  policy.use_working_memory=true policy.use_episodic_memory=true

# 评估
python eval_vis.py \
  --model_path experiments/LIBERO_SPATIAL/Multitask/MemFlowPolicy_seed0/run_001/multitask_model_ep500.pth \
  --task_id 0 --n_eval 5
```

---

## MemFlow vs Flow Matching 对比

| 特性 | Flow Matching | MemFlow |
|------|--------------|---------|
| **记忆机制** | 无 | L1 工作记忆 + L2 情景记忆 |
| **观测编码** | Policy Encoder | Policy Encoder + DINOv2（事件检测） |
| **条件向量** | obs_feat (256d) | obs_feat + working_mem + episodic_mem (768d) |
| **长期依赖** | ❌ 仅当前观测 | ✅ 追踪整个 episode 进度 |
| **动作连贯性** | ❌ 可能抖动 | ✅ 保持短期运动平滑 |
| **阶段感知** | ❌ 易混淆 | ✅ 区分"已做什么/该做什么" |
| **训练/推理一致性** | ✅ 天然一致 | ✅ **训练模拟推理的逐步处理** |

---

## 核心设计：训练/推理一致性

### 设计原则

**训练时模拟推理的逐步处理逻辑**，确保记忆更新方式完全一致。

```
┌────────────────────────────────────────────────────────────────┐
│                    训练：逐步模拟推理                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  演示序列: [obs_0, obs_1, ..., obs_T]                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ t = start_idx (第一个预测点)                             │   │
│  │   obs_feature_history = [feat_t]                         │   │
│  │   action_history = []                                    │   │
│  │   episodic_memory_bank = [dino_t] (首个事件)              │   │
│  │   → compute_condition() → predict action                 │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ t = start_idx + 1                                        │   │
│  │   obs_feature_history = [feat_0, feat_1]                 │   │
│  │   action_history = [gt_action_0]                         │   │
│  │   episodic_memory_bank 可能更新 (检测到事件)              │   │
│  │   → compute_condition() → predict action                 │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ ... 继续逐步处理，累积历史                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  收集所有 (v_pred, v_target) → MSE Loss                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 训练 vs 推理对比

| 方面 | 训练 | 推理 |
|------|------|------|
| **obs_feature 历史** | 逐步累积（模拟推理） | 逐步累积 |
| **action 历史** | 逐步累积（用 GT） | 逐步累积（用预测） |
| **事件检测** | 逐步单帧比较 | 逐步单帧比较 |
| **记忆条件计算** | 每步独立计算 | 每步独立计算 |
| **数据来源** | 演示数据 | 环境交互 |

**关键点**：记忆更新的逻辑完全相同，只有数据来源不同。

---

## 核心创新：层级记忆架构

### L1 工作记忆（Working Memory）—— 短期运动连贯性

```
输入: 最近 16 步的 (obs_feature, action) 序列
方法: Causal Transformer，交错编码 obs 和 action token
输出: 256 维记忆向量
作用: 保持动作连贯性，消除 chunk 边界处的突变抖动
时间尺度: ~1.6 秒 (@10Hz 控制频率)
```

**为什么需要 L1？**

Flow Policy 每次只看当前观测，缺乏历史上下文。在动作边界处容易产生：
- 位置跳变（上一帧输出 x=0.5，这一帧输出 x=0.3）
- 方向反转（上一帧向左，这一帧向右）
- 重复/遗漏动作

L1 工作记忆通过编码最近 16 步的运动轨迹，让策略"记住"刚才在干什么。

### L2 情景记忆（Episodic Memory）—— 中期任务进度追踪

```
输入: 观测序列 → DINOv2 提取特征序列
事件检测: 相邻帧 DINOv2 特征 L2 距离 > 阈值 → 标记为"关键事件"
存储: 稀疏 Memory Bank（一条 500 步轨迹只存 5-8 个事件）
读取: 当前观测作为 query → Cross-Attention 检索相关历史事件
输出: 256 维记忆向量
作用: 追踪"已经完成了哪些步骤"，消除视觉歧义
```

**为什么需要 L2？**

长 horizon 任务（如"依次拿 3 个物体放入盒子"）中：
- t=300 时，手是空的，桌上有 1 个物体
- 这和 t=0（拿第 1 个）视觉上很像
- 没有 L2 记忆：策略可能重复拿第 1 个物体
- 有 L2 记忆：知道"已放 2 个"，应该拿第 3 个

**为什么用 DINOv2？**

| 特性 | Policy Encoder | DINOv2 |
|------|----------------|--------|
| 训练目标 | 动作预测 | 自监督语义理解 |
| 对语义变化敏感 | ❌ 不一定 | ✅ 天然敏感 |
| 对像素变化敏感 | ✅ 可能过拟合 | ❌ 更鲁棒 |
| 事件检测稳定性 | ❌ 随训练变化 | ✅ 冻结，稳定 |

DINOv2 经过大规模预训练，天然能识别"抓取"、"放下"等语义变化，适合做事件检测。

### 两级记忆的协作

```
观测序列 obs[0:T]
    │
    ├─→ [DINOv2 冻结] → dino_feat_seq
    │       │
    │       ▼ detect_event_single()  ← 逐步检测
    │       │
    │       ▼ 累积 episodic_memory_bank
    │       │
    │       ▼ EpisodicMemory(current_dino, memory_bank)
    │       │
    │       ▼ e_t (256d)
    │
    └─→ [Policy Encoder + LSTM] → obs_feat_seq
            │
            ├─────────────────────────┐
            │                         │
            ▼                         ▼
       current_obs              action_history
       (B, 256)                  (B, T, 7)  ← 逐步累积
            │                         │
            └──────────┬──────────────┘
                       ▼
              WorkingMemory(obs_seq, action_seq)
                       │
                       ▼
                  m_t (256d)

条件向量: c_t = concat(current_obs, m_t, e_t) = 768d
```

**时间尺度分离**：

| 维度 | L1 工作记忆 | L2 情景记忆 |
|------|------------|------------|
| 时间尺度 | ~1.6 秒 | 整个 episode |
| 更新频率 | 每步更新 | 仅事件触发 |
| 存储内容 | 连续运动状态 | 离散关键事件 |
| 计算量 | O(K), K=16 | O(N_events), N≈5-8 |
| 解决的问题 | 动作平滑性 | 阶段追踪 |

---

## 配置参数

```yaml
# libero/configs/policy/memflow_policy.yaml

policy_type: MemFlowPolicy

# Flow Matching 基础参数
num_integration_steps: 10
horizon: 8
n_obs_steps: 2

# ============================================================
# 记忆模块开关
# ============================================================
use_working_memory: true    # L1: 开启短期运动连贯性
use_episodic_memory: true   # L2: 开启任务进度追踪

memory_dim: 256  # 记忆向量维度

# -----------------------------------------------------------
# L1 工作记忆参数
# -----------------------------------------------------------
working_memory_len: 16           # 记住最近 16 步
working_memory_hidden_dim: 256
working_memory_n_heads: 4
working_memory_n_layers: 2

# -----------------------------------------------------------
# L2 情景记忆参数
# -----------------------------------------------------------
use_dinov2_events: true          # 使用 DINOv2 事件检测
dinov2_model: dinov2-small       # 模型: dinov2-small (384d) 或 dinov2-base (768d)
dinov2_local_path: null          # 本地路径，设为 null 则自动下载

episodic_memory_hidden_dim: 256
episodic_memory_n_heads: 4
max_episodic_events: 32          # 最多存储 32 个事件
event_threshold_percentile: 95   # top 5% 距离判定为事件
```

---

## 安装依赖

```bash
# DINOv2 需要 transformers
pip install transformers
```

---

## DINOv2 预下载

DINOv2 模型约 86MB（small）或 330MB（base），首次运行会自动下载。也可以预下载：

```bash
# 下载到项目目录
python scripts/download_dinov2.py

# 下载到指定目录
python scripts/download_dinov2.py --output_dir /path/to/models

# 下载 base 版本
python scripts/download_dinov2.py --model dinov2_base
```

下载后配置本地路径：

```bash
python train.py policy=memflow_policy \
    policy.dinov2_local_path=/home/ydj/article/LIBERO/dinov2_models/dinov2-small
```

---

## 训练

### 基础训练（不开启记忆）

```bash
export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0

python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
    lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500
```

### 开启 L1 工作记忆

```bash
python train.py policy=memflow_policy \
    policy.use_working_memory=true \
    benchmark_name=LIBERO_SPATIAL \
    lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500
```

### 开启 L2 情景记忆

```bash
python train.py policy=memflow_policy \
    policy.use_episodic_memory=true \
    benchmark_name=LIBERO_SPATIAL \
    lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500
```

### 完整配置（两个记忆模块都开启）

```bash
python train.py policy=memflow_policy \
    policy.use_working_memory=true \
    policy.use_episodic_memory=true \
    policy.dinov2_local_path=dinov2_models/dinov2-small \
    benchmark_name=LIBERO_SPATIAL \
    lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=500
```

### 推荐消融实验

```bash
# 1. Baseline: Flow Matching
python train.py policy=flow_matching_policy benchmark_name=LIBERO_SPATIAL ...

# 2. MemFlow 无记忆（等价于 Flow Matching）
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL ...

# 3. MemFlow + L1 only
python train.py policy=memflow_policy policy.use_working_memory=true ...

# 4. MemFlow + L2 only
python train.py policy=memflow_policy policy.use_episodic_memory=true ...

# 5. MemFlow + L1 + L2 (Full)
python train.py policy=memflow_policy \
    policy.use_working_memory=true \
    policy.use_episodic_memory=true ...
```

---

## 评估

```bash
# 可视化评估
python eval_vis.py \
    --model_path experiments/LIBERO_SPATIAL/Multitask/MemFlowPolicy_seed0/run_001/multitask_model_ep500.pth \
    --task_id 0 \
    --n_eval 5

# 保存视频
python eval_vis.py \
    --model_path experiments/LIBERO_SPATIAL/Multitask/MemFlowPolicy_seed0/run_001/multitask_model_ep500.pth \
    --task_id 0 \
    --n_eval 5 \
    --save_video
```

---

## 模型架构

### 训练流程（逐步模拟推理）

```
训练序列: [obs_0, obs_1, ..., obs_T]
                  │
                  ▼ 先编码完整序列
         obs_feature_seq: (B, T, 256)
         dino_feature_seq: (B, T, 384)
                  │
                  ▼ 逐步处理
┌─────────────────────────────────────────────────────────────┐
│ for t in range(start_idx, T - horizon + 1):                 │
│                                                              │
│   1. 累积 obs_feature_history (供 Working Memory)            │
│      obs_feature_history.append(obs_feature_seq[:, t, :])   │
│                                                              │
│   2. 累积 action_history (供 Working Memory)                 │
│      if t > start_idx:                                       │
│          action_history.append(actions[:, t-1, :])  # GT    │
│                                                              │
│   3. 更新 Episodic Memory (逐步事件检测)                     │
│      current_dino = dino_feature_seq[:, t, :]               │
│      if is_event(current_dino, prev_dino):                  │
│          episodic_memory_bank.append(current_dino)          │
│                                                              │
│   4. 计算条件向量                                            │
│      cond = compute_memory_condition(                       │
│          obs_feature_history,                               │
│          action_history,                                    │
│          current_dino                                       │
│      )                                                       │
│                                                              │
│   5. Flow Matching 预测                                      │
│      target = actions[:, t:t+horizon, :]                    │
│      loss += MSE(v_pred, v_target)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 推理流程

```
环境交互 → 每步:
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 累积 obs_history (维护 n_obs_steps 帧)                    │
│    obs_history[img_name].append(current_obs)                │
│                                                              │
│ 2. 编码观测序列                                              │
│    obs_feature_seq = encode_obs_sequence(history_data)      │
│    obs_feature_history.append(current_obs_feature)          │
│                                                              │
│ 3. 使用 action_history (之前预测的动作)                      │
│    action_seq = torch.stack(action_history)                 │
│                                                              │
│ 4. 更新 Episodic Memory                                      │
│    current_dino = extract_dino(current_obs)                 │
│    update_episodic_memory_inference(current_dino)           │
│                                                              │
│ 5. 计算条件向量                                              │
│    cond = compute_memory_condition(...)                     │
│                                                              │
│ 6. Euler 积分生成动作                                        │
│    action = euler_integration(cond)                         │
│    action_history.append(action)                            │
│                                                              │
│ 7. 执行动作，获取新观测                                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键组件

| 组件 | 作用 | 输出维度 |
|------|------|---------|
| `encode_obs_sequence()` | LSTM 编码观测序列 | (B, T, 256) |
| `extract_dino_features_from_data()` | DINOv2 提取语义特征 | (B, T, 384) |
| `detect_event_single()` | 单帧事件检测 | bool |
| `update_episodic_memory_inference()` | 增量更新事件库 | - |
| `compute_memory_condition()` | 记忆条件计算 | (B, 768) |
| **WorkingMemory** | Causal Transformer | (B, 256) |
| **EpisodicMemory** | Cross-Attention | (B, 256) |
| **UNet 1D** | 向量场预测 | (B, horizon, 7) |

---

## 事件检测机制

### 训练时（逐步检测）

```python
# 在 forward() 循环中
for t in range(start_idx, T):
    current_dino = dino_feature_seq[:, t, :]

    # 与上一帧比较
    if prev_dino_feature is not None:
        distance = ||current_dino - prev_dino||
        is_event = distance > threshold

    if is_event:
        episodic_memory_bank.append(current_dino)

    prev_dino_feature = current_dino
```

### 推理时（在线检测）

```python
def update_episodic_memory_inference(self, dino_feature):
    # 单帧事件检测（自适应阈值）
    is_event, distance = self.dino_extractor.detect_event_single(
        dino_feature,
        self.prev_dino_feature,
        threshold_percentile=95
    )

    if is_event:
        self.episodic_memory_bank.append(dino_feature)

    self.prev_dino_feature = dino_feature
```

### 事件示例

```
Episode: "依次拿 3 个物体放入盒子"

帧   事件            DINOv2 距离   判定
0    初始状态         -            ✅ 事件（首帧）
45   抓住第1个物体    0.72         ✅ 事件
98   放下第1个物体    0.68         ✅ 事件
156  抓住第2个物体    0.65         ✅ 事件
203  放下第2个物体    0.71         ✅ 事件
278  抓住第3个物体    0.69         ✅ 事件
330  放下第3个物体    0.74         ✅ 事件

存储: 7 个事件（稀疏，而非 330 帧）
Cross-Attention 复杂度: O(7) vs O(330)
```

---

## 时间与资源估算

**LIBERO_SPATIAL（10 任务）**：

| 配置 | 训练时间 | GPU 显存 | 推理速度 |
|------|---------|---------|---------|
| 无记忆 | ~4 小时 | ~6GB | ~0.05s/action |
| +L1 | ~4.5 小时 | ~7GB | ~0.06s/action |
| +L2 | ~5 小时 | ~8GB | ~0.08s/action |
| +L1+L2 | ~5.5 小时 | ~9GB | ~0.09s/action |

**DINOv2 推理开销**：~50ms/frame（仅在 L2 开启时）

---

## 与论文设计的对应

| 论文设计 | 实现状态 | 位置 |
|---------|---------|------|
| L1 工作记忆（Causal Transformer） | ✅ 已实现 | `WorkingMemory` 类 |
| L2 情景记忆（事件驱动 + Cross-Attn） | ✅ 已实现 | `EpisodicMemory` 类 |
| DINOv2 事件检测 | ✅ 已实现 | `DINOv2FeatureExtractor` 类 |
| 训练/推理一致性 | ✅ 已实现 | `forward()` 逐步模拟推理 |
| 自适应步数路由（C2） | ❌ 待实现 | - |

当前实现对应论文创新点 C1（层级记忆架构）。C2（自适应步数路由）计划在后续版本实现。

---

## 常见问题

### Q: DINOv2 下载失败怎么办？

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_dinov2.py
```

### Q: 显存不足怎么办？

```bash
# 减小 batch size
python train.py ... train.batch_size=16

# 使用更小的 DINOv2
python train.py ... policy.dinov2_model=dinov2-small

# 关闭某个记忆模块
python train.py ... policy.use_working_memory=false
```

### Q: 事件检测太频繁/太少？

调整 `event_threshold_percentile`：
- 更高（如 98）→ 更少事件
- 更低（如 90）→ 更多事件

### Q: 为什么训练要模拟推理？

训练-推理不一致会导致：
1. 记忆模块在推理时的累积方式和训练不同
2. 模型无法有效利用记忆信息
3. 性能下降

MemFlow 通过训练时逐步累积历史（模拟推理），确保记忆更新逻辑完全一致。

---

## 参考文献

- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al., ICLR 2023)
- **DINOv2**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
- **Diffusion Policy**: [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- **LIBERO 基准**: [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
