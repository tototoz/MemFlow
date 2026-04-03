# Installtion
Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

 |
# MemFlow: Memory-Augmented Flow Policy with Adaptive Integration for Long-Horizon Robot Manipulation

## 1. 研究背景

### 1.1 问题背景

扩散/流匹配策略（Diffusion/Flow Matching Policy）已成为机器人模仿学习的主流范式，在抓取、推动、装配等任务上取得了显著成果。然而，现有方法存在两个根本性缺陷：

**缺陷一：无记忆性（Memoryless）**

现有策略基于当前观测（或极短的历史窗口）独立生成每一个动作块（action chunk），缺乏对任务历史的长期记忆。这导致：

- **重复动作**：分拣任务中，策略不记得哪些物体已经处理过，反复抓取同一物体
- **阶段混淆**：视觉上相似的场景（如"抽屉关着等待打开" vs "抽屉关好任务完成"）产生错误决策
- **失败后无法调整**：策略不知道上一次尝试失败了，会重复完全相同的错误动作
- **顺序约束违反**：不记得先后顺序，可能跳过必要步骤

**缺陷二：固定采样步数（Fixed Integration Steps）**

无论动作复杂度如何，现有方法对所有决策使用相同的去噪/积分步数：

- 匀速直线移动（分布简单）：2 步就够，但实际用了 10 步 → 浪费 80% 算力
- 精细抓取对准（分布复杂）：需要 16 步，但只给了 10 步 → 精度不足
- 固定步数是全局平均最优，而非每个样本的个体最优

### 1.2 核心洞察

**这两个缺陷是相互关联的**。如果策略知道自己处于任务的哪个阶段（记忆），就能判断当前动作的复杂度（难度），从而选择合适的积分步数（自适应加速）。

```
记忆 → 阶段感知 → 难度判断 → 自适应步数
```

因此，一个统一的记忆框架可以同时解决两个问题。

---

## 2. 创新点分析

### 2.1 创新点总览

| 编号 | 创新点 | 解决的问题 | 对应贡献 |
|------|--------|-----------|---------|
| C1 | 面向 Flow Policy 的层级记忆架构（工作记忆 + 情景记忆） | 长 horizon 任务阶段混淆、重复动作 | 性能提升 |
| C2 | 记忆引导的自适应积分步数路由 | 推理速度瓶颈 | 效率提升 |
| C3 | 事件驱动的稀疏记忆写入机制 | 记忆存储效率与计算开销 | 工程贡献 |
| C4 | 统一框架：记忆同时驱动决策和加速 | 两个独立问题的关联性 | 理论贡献 |

### 2.2 创新点 C1：层级记忆架构

**L1 工作记忆（Working Memory）—— 短期运动连贯性**

- 输入：最近 16 步的（观测特征，动作）序列
- 方法：Causal Transformer，交错编码 obs 和 action token
- 输出：256 维记忆向量
- 作用：保持动作连贯性，消除 chunk 边界处的突变抖动
- 时间尺度：~1.6 秒（@10Hz 控制频率）

**L2 情景记忆（Episodic Memory）—— 中期任务进度追踪**

- 核心机制：不存所有帧，只存"关键事件"
- 事件定义：相邻帧 DINOv2 特征差异超过阈值的时刻（如抓到物体、放下物体）
- 存储：稀疏 Memory Bank，一条 500 步轨迹只存 5-8 个事件
- 读取：当前观测作为 query，cross-attention 检索相关历史事件
- 作用：追踪"已经完成了哪些步骤"，消除视觉歧义

**为什么分两级？**

| 维度 | L1 工作记忆 | L2 情景记忆 |
|------|------------|------------|
| 时间尺度 | ~1.6 秒 | 整个 episode |
| 更新频率 | 每步更新 | 仅事件触发 |
| 存储内容 | 连续运动状态 | 离散关键事件 |
| 计算量 | O(K), K=16 | O(N_events), N≈5-8 |
| 解决的问题 | 动作平滑性 | 阶段追踪 |

如果用一个统一的 memory 处理两种信息，短期细粒度运动信号会淹没长期稀疏的阶段信号。分级设计让每一级专注于自己的时间尺度。

### 2.3 创新点 C2：记忆引导的自适应步数路由

**Router 输入：记忆状态（L1 + L2 + 当前观测特征）**

**Router 输出：积分步数 ∈ {2, 4, 8, 16}**

**与 ProbeFlow（最直接的竞争者）的本质区别：**

| 对比维度 | ProbeFlow | Ours |
|----------|-----------|------|
| 判断依据 | 速度场余弦相似度（几何曲率） | 记忆状态（任务语义） |
| 判断时机 | 事后：先试一步再看弯不弯 | 事前：看记忆直接预判 |
| 额外开销 | 2 次前向传播（探测用） | 1 次轻量 MLP（~0.1ms） |
| 语义感知 | 无（纯数学指标） | 有（知道任务上下文） |
| 失败案例 | 轨迹前半段直后半段弯 → 误判 | 记忆知道即将进入精细阶段 → 正确 |

**记忆如何帮助步数预测的具体例子：**

```
场景：机器人桌面整理，需要依次拿 3 个物体放入盒子

时刻 t=200: 正在搬运第 2 个物体（匀速移动）
  只看当前帧: 手里有物体，不确定在干什么
  ProbeFlow:  速度场恰好是直的 → 选 2 步 ✅（碰巧对了）

时刻 t=250: 准备放下第 2 个物体（接近盒子上方）
  只看当前帧: 手在盒子上方，和放第 1 个时一样
  ProbeFlow:  速度场开始弯曲 → 选 8 步 ✅

时刻 t=300: 放完第 2 个，转头去拿第 3 个
  只看当前帧: 手是空的，桌上有 1 个物体 → 和 t=0（拿第 1 个）视觉很像
  ProbeFlow:  速度场方向取决于噪声初始化，不确定 → 可能选错
  Ours: 情景记忆记录了"已放 2 个" → 知道该拿第 3 个 → 明确的目标
        → 路径清晰 → 选 4 步 ✅
```

### 2.4 创新点 C3：事件驱动稀疏记忆

**为什么不存所有帧？**

- 500 步任务存所有帧：memory bank 500 条，cross-attention O(500) 每次决策
- 只存事件：bank 5-8 条，cross-attention O(5-8)，快约 100 倍
- 连续帧之间 >95% 信息冗余，事件帧之间信息增益最大

**事件检测方法：**

- 用冻结的 DINOv2 提取每帧特征
- 计算相邻帧特征的 L2 距离
- 距离超过阈值（取所有帧间距离的 top 5% 分位数）→ 标记为事件
- 标签离线预计算，不需要人工标注

**认知科学依据：**

事件驱动记忆与人类的 Event Segmentation Theory 一致——人类不记得吃饭时每一口的细节，但记得"开始吃""吃完了"这些转折点。我们的机制模拟了这种认知模式。

### 2.5 创新点 C4：统一框架

本文不是两个独立 trick 的简单拼凑，而是一个有内在逻辑的统一框架：

```
                    记忆模块
                   ╱        ╲
        解决阶段歧义          引导步数分配
        （性能提升）          （效率提升）
                   ╲        ╱
                  统一的条件向量 c_t
                       │
                       ▼
                  Flow Policy
```

消融实验 (e)（Router 只看 obs，不看记忆）vs Full 的对比是证明这一点的关键证据：记忆确实帮助了步数路由，不是可有可无的。

---

## 3. 方法详细设计

### 3.1 整体架构

```
输入: 当前观测 o_t + 历史 {(o_i, a_i)}_{i<t}

┌────────────── Stage 1: 感知 + 记忆 ──────────────┐
│                                                    │
│  o_t → [DINOv2 冻结] → f_t (观测特征, 256维)      │
│                            │                       │
│  历史 → [L1 Causal Transformer] → m_t (工作记忆)   │
│                            │                       │
│  历史 → [L2 Event Detect → Memory Bank             │
│              → Cross-Attn Read] → e_t (事件记忆)   │
│                                                    │
│  条件向量: c_t = concat(f_t, m_t, e_t)  (768维)   │
└──────────────────────┬─────────────────────────────┘
                       │
┌──────────────────────┼─── Stage 2: 自适应生成 ─────┐
│                      │                              │
│              ┌───────┴───────┐                      │
│              ▼               ▼                      │
│     [Step Router]     [Flow Policy]                 │
│     R_ψ(c_t) → n     v_θ(x, t, c_t)               │
│              │               │                      │
│              └───────┬───────┘                      │
│                      ▼                              │
│         Euler ODE Solve (n 步)                      │
│         x_{i+1} = x_i + v · (1/n)                  │
│                      │                              │
│                      ▼                              │
│                  action a_t                         │
└─────────────────────────────────────────────────────┘
```

### 3.2 观测编码器

- 使用冻结的 DINOv2-S（ViT-S/14）作为视觉 backbone
- 输出 384 维 CLS token，投影到 256 维
- 选择 DINOv2 的理由：
  - 对光照/背景/视角变化鲁棒（提升泛化）
  - 特征语义丰富（事件检测更准）
  - 冻结不训练（减少参数量和过拟合）

### 3.3 L1 工作记忆

- 输入：最近 K=16 步的 (obs_feat, action) 对
- 编码方式：obs 和 action 交替排列为 token 序列 [f_0, a_0, f_1, a_1, ..., f_15, a_15]
- 添加类型嵌入（区分 obs/action）和位置嵌入
- 通过 4 层 Causal Transformer（Pre-LN，4 heads）
- 取最后一个 token 的隐状态作为工作记忆 m_t ∈ R^256

### 3.4 L2 情景记忆

**事件检测：**
- 计算相邻帧 DINOv2 特征的 L2 距离
- 经过一个 2 层 MLP（输入：当前帧+上一帧特征的拼接；输出：sigmoid 门控值）
- 门控值 > 0.5 时写入 Memory Bank
- 训练时用 soft gate 允许梯度流过，推理时用 hard threshold

**记忆读取：**
- 当前观测特征 f_t 作为 query
- Memory Bank 中的事件特征作为 key 和 value
- 4-head cross-attention + 残差连接
- 输出事件记忆 e_t ∈ R^256

**容量管理：**
- Memory Bank 最大容量 20 个事件
- 超过时 FIFO（先进先出）策略丢弃最早的事件

### 3.5 自适应步数路由器

- 输入：c_t = [f_t; m_t; e_t]（768 维）
- 网络：3 层 MLP（768 → 256 → 128 → 4），带 LayerNorm 和 GELU
- 输出：4 个 logits，对应步数 {2, 4, 8, 16}
- 训练时用 Gumbel-Softmax（温度从 5.0 退火到 0.5）
- 推理时用 argmax
- 参数量：~50K（占总模型 <0.15%）

### 3.6 Flow Matching Policy

- 标准条件流匹配（Conditional Flow Matching）
- 速度场网络：ConditionalUNet1D（与 Diffusion Policy 结构一致）
- 条件维度：768（obs 256 + memory 256 + event 256）
- 动作空间：7 维（末端执行器 6D pose + gripper）
- 预测 horizon：16 步 action chunk
- 与标准 Flow Matching Policy 的唯一区别是条件输入维度（256 → 768）

### 3.7 训练流程

**Phase 1：主训练（记忆 + Flow Policy）**

- 同时训练：DINOv2 投影层 + 记忆编码器 + 事件检测器 + Flow Policy
- 固定积分步数 = 8（Router 暂不训练）
- 损失函数：
  - L_flow = E[||v_θ(x_t, t, c_t) - (x_1 - x_0)||²]（标准 Flow Matching loss）
  - L_event = BCE(gate, event_label)（事件检测监督）
  - L_total = L_flow + α · L_event（α = 0.1）
- 训练量：~1500 epochs

**Phase 2：Router 训练**

- 冻结 Phase 1 的所有模块
- 只训练 Router（~50K 参数）
- 损失函数：
  - L_quality = 选中步数方案的动作误差
  - L_speed = 选中的步数值（越小越快）
  - L_entropy = -H(softmax(logits))（防止坍缩到单一选择）
  - L_router = L_quality + λ_speed · L_speed + λ_entropy · L_entropy
  - λ_speed = 0.1, λ_entropy = 0.01
- 训练量：~100 epochs

**Phase 3：联合微调**

- 解冻所有模块，端到端微调
- 小学习率（Phase 1 的 1/10）
- 训练量：~50 epochs

### 3.8 推理流程

```
每一个控制步:
  1. 编码当前观测 f_t = DINOv2(o_t)                   ~2ms
  2. 更新工作记忆 m_t = MemoryEncoder(history)         ~1ms
  3. 事件检测 + 更新 Memory Bank                       ~0.5ms
  4. 读取事件记忆 e_t = CrossAttn(f_t, bank)          ~0.3ms
  5. Router 选步数 n = Router(f_t, m_t, e_t)          ~0.1ms
  6. Flow Policy 采样 a_t = EulerODE(v_θ, c_t, n步)   n×1.5ms
  ─────────────────────────────────────────────────
  总计 (n=2):  ~7ms   → 143Hz
  总计 (n=4):  ~10ms  → 100Hz
  总计 (n=8):  ~16ms  → 63Hz
  总计 (n=16): ~28ms  → 36Hz
  加权平均 (35%/30%/25%/10%): ~11ms → 91Hz
```

---

## 4. 实验设计

### 4.1 数据集与环境

| 数据集 | 任务类型 | 任务数 | Demo 数量 | 用途 |
|--------|---------|--------|----------|------|
| LIBERO-Spatial | 空间关系变化（同物体不同位置） | 10 | 50 条/任务 | 全面对比（空间歧义消解） |
| LIBERO-Object | 物体种类变化（同场景不同目标） | 10 | 50 条/任务 | 全面对比（物体识别） |
| LIBERO-Goal | 任务目标变化（多种操作目标） | 10 | 50 条/任务 | **目标消歧核心实验** |
| LIBERO-100 | 综合多样化 | 100 | 50 条/任务 | **大规模多任务核心实验** |

**数据集下载：**

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py
```

**实验策略：**

- **核心实验 1**：LIBERO-Goal（任务目标各不同，视觉场景相似，最依赖记忆消歧）
- **核心实验 2**：LIBERO-100（100 个多样化任务，验证大规模多任务可扩展性）
- **全面对比**：LIBERO-Spatial + LIBERO-Object（证明记忆在所有子集上均不退化）

---

### 4.2 Baseline 方法

| 方法 | 类别 | 积分步数 | 说明 | 实现状态 |
|------|------|---------|------|---------|
| Diffusion Policy (DDIM-10) | 扩散 | 10 固定 | 最经典 baseline | ✅ 已集成，训练中 |
| Flow Matching Policy (Euler-10) | 流匹配 | 10 固定 | 无记忆的 Flow Policy 直接前身 | ⬜ 待复现 |
| Consistency Policy (3步) | 蒸馏加速 | 3 固定 | 一致性蒸馏快速推理 | ⬜ 待复现 |
| ProbeFlow | 自适应流匹配 | 2-10 自适应 | 几何曲率启发式，最直接竞争者 | ⬜ 待复现 |
| ACT | Transformer | N/A | Action Chunking Transformer | ⬜ 待复现 |

---

### 4.3 评估指标

#### LIBERO 指标

| 指标 | 符号 | 计算方式 | 说明 |
|------|------|---------|------|
| 成功率 | SR | 每任务 20 episodes 成功数 / 20 | 主要指标 |
| 平均成功率 | Avg SR | 所有任务 SR 的均值 | 汇总指标 |
| 前向迁移 | FWT | 学完任务 i 后在任务 i 上的 SR | 学习效率 |
| 后向迁移 | BWT | 学完所有任务后在任务 i 上的 SR − 学完任务 i 时的 SR | 遗忘程度（越接近0越好） |

> Multitask 训练下只报 SR 和 Avg SR；Sequential 训练下额外报 BWT。

#### 推理效率指标

| 指标 | 单位 | 说明 |
|------|------|------|
| 推理延迟 | ms/step | 观测→动作端到端时间（GPU，batch=1） |
| 平均积分步数 | steps | Router 实际选择步数均值（MemFlow 专属） |
| 步数分布 | % | 各档（2/4/8/16步）选择频率 |

---

### 4.4 实验列表

#### 实验 1：主实验对比（论文 Table 1）

**目的**：证明 MemFlow 在所有 LIBERO 子集上全面优于 baseline。

**评测规模**：每方法 × 每数据集 × 3 seeds，Multitask 训练，每任务 20 episodes。

**Table 1 结构（LIBERO 成功率，均值±标准差）：**

```
方法               │ LIBERO-Spatial │ LIBERO-Object │ LIBERO-Goal │ LIBERO-100 │ 推理延迟
───────────────────│────────────────│───────────────│─────────────│────────────│────────
Diffusion Policy   │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
Flow Matching      │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
Consistency Policy │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
ProbeFlow          │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
ACT                │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
───────────────────│────────────────│───────────────│─────────────│────────────│────────
MemFlow (ours)     │  xx.x ± x.x   │  xx.x ± x.x  │ xx.x ± x.x │ xx.x ± x.x │  xxms
```

> LIBERO-Goal 和 LIBERO-100 是体现记忆价值最明显的两列，应在正文中重点分析。

---

#### 实验 2：消融实验（论文 Table 2）

**目的**：证明每个模块的必要性，核心是"记忆引导 Router"的贡献。

**在 LIBERO-Goal + LIBERO-100 上各跑，3 seeds：**

| 变体 | L1 工作记忆 | L2 情景记忆 | Router | 记忆→Router |
|------|-----------|-----------|--------|------------|
| (a) MemFlow 完整 | ✓ | ✓ | ✓ | ✓ |
| (b) w/o L2 情景记忆 | ✓ | ✗ | ✓ | 部分 |
| (c) w/o L1 工作记忆 | ✗ | ✓ | ✓ | 部分 |
| (d) w/o Router，固定 8 步 | ✓ | ✓ | ✗ | — |
| (e) Router 不看记忆，只看 obs | ✓ | ✓ | ✓ | ✗ |
| (f) 标准 Flow Policy | ✗ | ✗ | ✗ | — |

> **(e) vs (a) 是论文最关键的消融**：证明记忆不仅提升生成质量，还明确帮助了推理效率。
> **(d) vs (a)** 证明 Router 带来的加速不以牺牲性能为代价。

---

#### 实验 3：速度-质量 Pareto 曲线（论文 Figure）

**目的**：可视化所有方法在"推理延迟 vs 成功率"二维平面的位置，展示 MemFlow 位于 Pareto 前沿。

- X 轴：推理延迟（ms/step）
- Y 轴：LIBERO-100 Avg SR（任务最多，最有区分度）
- MemFlow 用不同 λ_speed 画成一条曲线，展示可灵活调节的速度-质量 tradeoff

**λ_speed 敏感性扫参（附表 Table 3）：**

| λ_speed | 平均步数 | LIBERO-100 SR | LIBERO-Goal SR | 推理延迟 |
|---------|---------|--------------|----------------|---------|
| 0.01 | ~14 | xx.x% | xx.x% | ~26ms |
| 0.05 | ~10 | xx.x% | xx.x% | ~19ms |
| **0.1（默认）** | **~7** | **xx.x%** | **xx.x%** | **~14ms** |
| 0.2 | ~5 | xx.x% | xx.x% | ~11ms |
| 0.5 | ~3 | xx.x% | xx.x% | ~8ms |

---

#### 实验 4：Router 步数分布分析（论文 Figure）

**目的**：展示 Router 学到了有语义意义的步数分配规律。

- 在 LIBERO-100 任务轨迹上逐步记录 Router 选择的步数
- 按任务阶段标注（接近物体 / 抓取 / 搬运 / 精细放置）
- 绘制时间轴上的步数选择热图

**预期结论**：
- 匀速搬运阶段 → 主要选 2-4 步
- 精细对准/放置阶段 → 主要选 8-16 步
- 步数分布与任务语义强相关，Router 学到了有意义的规律

---

#### 实验 5：事件记忆可视化（论文 Figure）

**目的**：定性验证 L2 情景记忆存储的是真正有语义意义的关键帧。

- 展示一条完整轨迹（~200 帧）的 DINOv2 特征差 L2 距离变化曲线
- 高亮写入 Memory Bank 的帧（峰值处）
- 截取关键帧图像，人工核验语义

**预期**：Memory Bank 中存的是"开始抓取"、"成功抓起"、"开始放置"、"放置完成"等转折点，而非随机帧。

---

#### 实验 6：记忆向量 t-SNE（论文 Figure，可选）

**目的**：展示工作记忆 m_t 具有清晰的任务阶段语义聚类结构。

- 收集多条轨迹的 m_t 向量，按阶段（接近/抓取/搬运/放置）着色
- t-SNE 降至 2D 可视化
- 预期：同阶段向量聚在一起，证明记忆编码了任务进度

---

### 4.5 创新点与实验对应分析

根据实验设计，所有 4 个创新点均有充分的实验支撑。以下是详细的对应关系分析：

#### C1（层级记忆架构）→ 多层次验证

**实验 2 消融 + 实验 6 可视化**

| 消融变体 | 验证内容 | 对应关系 |
|---------|---------|---------|
| (b) w/o L2 情景记忆 | L2 的独立贡献 | 证明 L2 的必要性（预期 LIBERO-Goal 下降 5-8%） |
| (c) w/o L1 工作记忆 | L1 的独立贡献 | 证明 L1 的必要性（预期动作连贯性下降） |
| (a) Full | L1+L2 协同效应 | 证明分级设计优于单一记忆 |

**补充证据**：实验 6 的 t-SNE 可视化能直观展示工作记忆 m_t 在不同任务阶段形成清晰聚类，证明记忆编码了任务进度语义。

**结论**：✅ 充分。消融实验 能直接证明两级记忆各有独立贡献，不是冗余设计。

---

#### C2（记忆引导 Router）→ 关键消融 + 分布分析

**实验 2 消融 + 实验 4 步数分布**

**(e) vs (a) 是论文最核心的消融实验**：

```
变体                     │ Router 输入      │ 预期结果
─────────────────────────│─────────────────│───────────
(a) MemFlow 完整        │ concat(f, m, e)  │ SR ~80%, 平均步数 ~7, 延迟 ~11ms
(e) Router 不看记忆     │ 仅 f_t (obs)     │ SR ~77%, 平均步数 ~9, 延迟 ~16ms
```

**为什么 能证明 C2？**
- 如果 Router 不看记忆，既降低了质量（SR -3%）又降低了效率（平均步数 +2），说明记忆确实提供了有价值的语义信息
- 这证明了创新点 C2 的核心主张：记忆 → 阶段感知 → 步数预测

**实验 4 补充证据**：
- 展示 Router 在"匀速搬运阶段"主要选 2-4 步，在"精细对准阶段"主要选 8-16 步
- 证明 Router 学到了有语义意义的规律，而非随机分配

**结论**：✅ 非常充分。消融 是直击 C2 的关键实验，加上实验 4 的语义分析，形成完整的证据链。

---

#### C3（事件驱动稀疏记忆）→ 可视化验证

**实验 5 事件记忆可视化**

虽然实验 5 是定性分析，但对 C3 非常关键：

- 展示 Memory Bank 存的是"抓取开始"、"放置完成"等语义转折点
- 证明事件检测机制有效，而非随机存帧

**可选补充**：
在实验 2 中增加一个变体：
```
(g) L2 存所有帧（非事件驱动）
    → 预期：推理延迟 +30%（cross-attention O(500) vs O(5-8)），性能相近或略降
```

**结论**：✅ 基本充分。可视化能定性验证 C3，但缺少定量效率对比。如果时间紧张，当前设计已足够；如果时间充裕，建议补充 的效率消融。

---

#### C4（统一框架）→ 多实验综合证明

**实验 1 + 实验 2 + 实验 3 联合证明**

C4 的核心主张是"记忆同时解决质量（C1）和效率（C2）两个问题"，需要多个实验共同支撑：

| 实验 | 证明内容 | C4 证据 |
|------|---------|---------|
| 实验 1 主对比 | MemFlow 质量不劣于 baseline | SR ~80% vs DP ~73% |
| 实验 2 消融 | 记忆单独贡献（d vs f）：+8% SR | 质量提升来自记忆 |
| 实验 3 Pareto | MemFlow 位于前沿，质量和速度双优 | 唯一在 ~11ms 达到 ~80% SR 的方法 |

**关键证据链**：
```
(d) 记忆+固定8步 → SR ~81%, 延迟 ~16ms
(a) 记忆+Router  → SR ~80%, 延迟 ~11ms
```
- 这证明 Router 没有牺牲质量（SR -1%），但显著提升了效率（延迟 -5ms）
- 结合实验 2 消融，证明记忆是质量和效率共同提升的根源

**结论**：✅ 充分。实验 3 的 Pareto 曲线是 C4 的最佳可视化证明，配合消融 形成完整论证。

---

#### 总体评估

| 创新点 | 支持实验 | 充分性 | 建议补充 |
|--------|---------|--------|---------|
| C1（层级记忆） | 实验 2 消融 + 实验 6 可视化 | ✅ 充分 | — |
| C2（记忆引导 Router） | 实验 2 消融 + 实验 4 分布 | ✅ 非常充分 | — |
| C3（事件驱动稀疏记忆） | 实验 5 可视化 | ✅ 基本充分 | 可选：消融 vs 存所有帧 |
| C4（统一框架） | 实验 1+2+3 综合 | ✅ 充分 | — |

---

### 4.6 预期结果与论文叙事逻辑

**预期数字：**

```
方法                         │ LIBERO-Spatial │ LIBERO-Object │ LIBERO-Goal │ LIBERO-100 │ 推理延迟
─────────────────────────────│────────────────│───────────────│─────────────│────────────│────────
Diffusion Policy (DDIM-10)   │     ~73%       │     ~75%      │    ~70%     │    ~60%    │  ~17ms
Flow Matching Policy (10步)  │     ~75%       │     ~77%      │    ~72%     │    ~62%    │  ~17ms
Consistency Policy (3步)     │     ~68%       │     ~70%      │    ~65%     │    ~55%    │   ~6ms
ACT                          │     ~74%       │     ~76%      │    ~71%     │    ~61%    │   ~5ms
ProbeFlow                    │     ~76%       │     ~78%      │    ~73%     │    ~63%    │  ~10ms
─────────────────────────────│────────────────│───────────────│─────────────│────────────│────────
MemFlow 记忆+固定8步          │     ~81%       │     ~83%      │    ~80%     │    ~72%    │  ~16ms
MemFlow 记忆+自适应（默认）   │     ~80%       │     ~82%      │    ~79%     │    ~71%    │  ~11ms
```

> 记忆在 LIBERO-Goal（目标消歧）和 LIBERO-100（大规模多任务）上提升最大，这是论文叙事的重点。

**论文结论链（实验结果应依次支持以下 5 点）：**

1. **记忆有效**（Table 2：(d) vs (f)）：加记忆后 LIBERO-Goal SR +8%，LIBERO-100 SR +10%
2. **L1+L2 缺一不可**（Table 2：(b)(c) vs (a)）：两级记忆各有独立贡献
3. **记忆帮助了 Router**（Table 2：(e) vs (a)）：记忆引导比只看 obs 的 Router 更快更准
4. **Pareto 前沿**（Figure 速度-质量曲线）：MemFlow 是唯一在质量不退化情况下达到 ~11ms 的方法
5. **可解释**（Figure：Router 分析 + 事件可视化）：Router 决策有语义规律，Memory Bank 存的是真正的关键事件

---

### 4.6 实验规模估算

| 实验 | GPU 需求 | 时间估算 |
|------|---------|---------|
| 5 种 baseline × 4 LIBERO × 3 seeds | 8× A100 | ~3 天（并行） |
| MemFlow Phase 1 训练（LIBERO × 4） | 2× A100 | ~1.5 天 |
| MemFlow Phase 2+3 训练 | 1× A100 | ~0.5 天 |
| 消融实验（6 变体 × LIBERO-Goal + LIBERO-100） | 4× A100 | ~1.5 天 |
| λ_speed 扫参（5组）+ 可视化分析 | 1× A100 | ~半天 |
| **总计** | **8× A100** | **~7 天** |

---

## 5. 论文结构

```
Title: MemFlow: Memory-Augmented Flow Policy with Adaptive Integration
       for Long-Horizon Robot Manipulation

Abstract (250 words)

1. Introduction (1.5 pages)
   - 问题：Flow Policy 无记忆 + 固定步数
   - 洞察：两个问题通过"阶段感知"关联
   - 贡献：统一的记忆框架同时解决两个问题

2. Related Work (1 page)
   2.1 Diffusion/Flow Matching Policy (DP, FMP, π₀, ACT)
   2.2 采样加速方法 (ProbeFlow, Consistency Policy, A2A, SDP, DDIM)
   2.3 记忆与机器人学习 (Memory-augmented IL, episodic memory in RL)
   2.4 自适应计算 (CALM, early exit, adaptive depth in NLP/CV)

3. Method (3 pages)
   3.1 Preliminaries: Conditional Flow Matching
   3.2 Working Memory (L1)
   3.3 Episodic Memory (L2) + Event Detection
   3.4 Memory-Guided Adaptive Step Router
   3.5 Training Pipeline (Phase 1 → 2 → 3)
   3.6 Inference

4. Experiments (3 pages)
   4.1 Setup (数据集, Baseline, 评测协议)
   4.2 Main Results (Table 1, 2)
   4.3 Ablation Study (Table 3)
   4.4 Router Behavior Analysis (Figure)
   4.5 Comparison with ProbeFlow (Figure + Case Study)
   4.6 Visualization (Figure)

5. Conclusion (0.5 page)

Appendix:
   A. Implementation Details
   B. Additional Experiments
   C. Hyperparameter Sensitivity
```

目标会议：CoRL 2026 / RSS 2027

---

## 6. 项目计划

### 6.0 当前进展（2026-04）

| 模块 | 状态 | 说明 |
|------|------|------|
| LIBERO 环境搭建 | ✅ 完成 | conda 环境、数据集、基准均已就绪 |
| Diffusion Policy 集成 | ✅ 完成 | 已集成到 LIBERO 框架，可正常训练 |
| Diffusion Policy 训练 | 🔄 进行中 | `LIBERO_SPATIAL`，`seed=42`，`eval.eval=false` |
| Flow Matching Policy | ⬜ 待开始 | — |
| MemFlow 主模块 | ⬜ 待开始 | — |

**当前代码结构（已落地部分）：**

```
LIBERO/                           ← LIBERO benchmark 框架
├── policy/
│   └── diffusion_policy/         ← Diffusion Policy baseline（已完成）
├── train.py                      ← 统一训练入口（已修改，支持 eval.eval=false）
├── eval_vis.py                   ← MuJoCo 实时可视化评估脚本
└── experiments/                  ← 训练权重输出目录
    └── LIBERO_SPATIAL/Sequential/DiffusionPolicy_seed42/run_001/
        ├── task0_model.pth ~ task9_model.pth
```

**Diffusion Policy 训练命令：**

```bash
export CUDA_VISIBLE_DEVICES=0 && export MUJOCO_EGL_DEVICE_ID=0
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL seed=42 'eval.eval=false'
```

---

### 6.1 团队分工

```
成员 A — 记忆模块负责人
  职责：L1 工作记忆 + L2 情景记忆 + 事件检测器
  技能要求：Transformer 实现经验、注意力机制

成员 B — 加速模块 + 训练负责人
  职责：Flow Policy 主干 + Router + 训练流程（Phase 1-3）
  技能要求：Flow Matching 理论、训练调优

成员 C — 实验 + 论文负责人
  职责：环境搭建 + 全部 baseline + 评测 + 可视化 + 统稿
  技能要求：LIBERO 经验、论文写作
```

### 6.2 四周计划

```
Week 1 ─── 并行启动

  A:
    ├── Day 1-3: 实现 L1 Causal Transformer 记忆编码器
    ├── Day 4-5: 实现 L2 事件检测器 + Memory Bank + Cross-Attn 读取
    └── Day 6-7: 单元测试，确认模块输入输出正确

  B:
    ├── Day 1-3: 实现 Flow Matching Policy 主干代码
    ├── Day 4-5: 实现 Router 网络 + 多步数 Euler 采样
    ├── Day 6: 预计算事件标签（DINOv2 特征差，离线批处理）
    └── Day 7: 定义好 A/B 模块的接口规范

  C:
    ├── ✅ Day 1-2: 搭建 LIBERO 环境，下载数据
    ├── ✅ Day 3-4: Diffusion Policy baseline 集成并验证训练流程
    ├── Day 5-7: 并行训练所有 baseline（多卡同时跑）
    │   ├── GPU 0-1: Flow Matching Policy
    │   ├── GPU 2-3: Diffusion Policy（已完成集成，正在训练）
    │   ├── GPU 4-5: Consistency Policy
    │   └── GPU 6-7: ProbeFlow + ACT
    └── Day 7: 编写统一评测脚本

  ✅ Week 1 检查点: A+B 模块能拼通，C 有全部 baseline 数字


Week 2 ─── 集成 + 训练

  A+B 合并:
    ├── Day 1: 代码集成，解决接口问题
    ├── Day 2-4: Phase 1 训练（记忆 + Flow Policy）
    │            A 监控训练，调记忆超参
    │            B 同时完善 Phase 2 Router 训练代码
    ├── Day 5-6: Phase 2 Router 训练
    └── Day 7: Phase 3 联合微调

  C:
    ├── Day 1-4: 写 Related Work + Method 框架（数字用占位符）
    ├── Day 5-7: 准备全部实验评测脚本 + 可视化代码
    └── 整理 baseline 结果表格

  ✅ Week 2 检查点: MemFlow 在 LIBERO-Goal 上有明显提升（SR >75%）
  ⚠️ 如果没有提升: 三人一起排查，这是存亡时刻
     → 检查事件标签质量
     → 加大记忆窗口 K
     → 尝试 GRU 替代 Transformer


Week 3 ─── 全量实验（8 卡全开）

  A 负责消融实验:
    ├── (a) Full MemFlow
    ├── (b) w/o L2
    ├── (c) w/o L1
    ├── (d) w/o Router
    ├── (e) Router 不看记忆
    └── (f) 标准 Flow Policy

  B 负责主实验 + Router 分析:
    ├── LIBERO 四个子集: 3 seeds × 20 episodes
    ├── LIBERO-100: 3 seeds × 20 episodes/任务（共 100 任务）
    ├── λ_speed 敏感性扫参 {0.01, 0.05, 0.1, 0.2, 0.5}
    └── Router 步数分布统计

  C 负责补充实验 + 可视化 + 写作:
    ├── RoboMimic 补充实验
    ├── ProbeFlow 对比 case 分析
    ├── 可视化（t-SNE, 事件检测, Router 分布, Pareto 曲线）
    └── 持续写 Experiments（数字出一个填一个）

  ✅ Week 3 检查点: 所有实验数据收齐，论文完成 70%


Week 4 ─── 论文冲刺

  A:
    ├── Day 1-2: 写 Method 3.2 + 3.3（记忆模块细节）
    ├── Day 3: 补跑缺失实验
    └── Day 4-5: 交叉审稿

  B:
    ├── Day 1-2: 写 Method 3.4 + 3.5（Router + 训练流程）
    ├── Day 3-4: 画架构图（Figure 1 总览图, Figure 2 记忆细节图）
    └── Day 4-5: 交叉审稿

  C:
    ├── Day 1-2: 写 Introduction + Conclusion + Abstract
    ├── Day 3: 整合 A/B 的 Method，统一写作风格
    ├── Day 4: 格式排版、参考文献、Appendix
    ├── Day 5: 导师/内部审稿
    ├── Day 6: 根据反馈修改
    └── Day 7: 提交

  ✅ Week 4 检查点: 论文提交
```

### 6.3 里程碑与风险管理

| 里程碑 | 时间 | 成功标准 | 失败应对 |
|--------|------|---------|---------|
| M1: 模块拼通 | Week 1 末 | A+B 代码集成后能跑通前向传播 | Day 1 就定好接口，用 mock 数据测试 |
| M2: 记忆有效 | Week 2 Day 4 | LIBERO-Goal SR >75%（baseline ~70%） | 砍 L2 只保留 L1 降低复杂度；换 GRU |
| M3: Router 有效 | Week 2 Day 6 | 平均步数 <6 且性能下降 <2% | 加强熵正则；调 λ_speed |
| M4: 实验完整 | Week 3 末 | 所有表格和图的数据齐全 | 优先保证主实验 + 消融，补充实验可缩减 |
| M5: 论文提交 | Week 4 末 | 完整论文 PDF | 若来不及：先投 arXiv 占坑 |

### 6.4 硬件需求

```
训练:
  Phase 1: 2×A100 (80GB), ~12 小时
  Phase 2: 1×A100, ~3 小时
  Phase 3: 2×A100, ~4 小时
  Baseline 复现: 6×A100, ~2-3 天（并行）
  消融实验: 6×A100, ~1-2 天（并行）
  总计: 8×A100, 全量实验 ~500 GPU 小时

推理/评测:
  每组实验 (3 seeds × 20 episodes): ~6-10 小时/GPU
  LIBERO-100 评测 (3 seeds × 100 任务 × 20 eps): ~8 小时/GPU
```

### 6.5 代码结构

```
memflow/
├── configs/                  # 实验配置文件
│   ├── libero_spatial.yaml
│   ├── libero_object.yaml
│   ├── libero_goal.yaml
│   ├── libero_100.yaml
│   ├── calvin.yaml
│   └── robomimic.yaml
├── models/
│   ├── obs_encoder.py        # DINOv2 观测编码器
│   ├── working_memory.py     # L1 Causal Transformer
│   ├── episodic_memory.py    # L2 事件检测 + Memory Bank
│   ├── flow_policy.py        # Flow Matching Policy
│   ├── router.py             # 自适应步数路由器
│   └── memflow.py            # 完整模型（整合所有模块）
├── training/
│   ├── phase1_train.py       # 记忆 + Flow Policy 训练
│   ├── phase2_router.py      # Router 训练
│   ├── phase3_finetune.py    # 联合微调
│   └── event_label_gen.py    # 离线事件标签预计算
├── evaluation/
│   ├── eval_libero.py
│   ├── eval_calvin.py
│   └── eval_robomimic.py
├── visualization/
│   ├── router_distribution.py
│   ├── tsne_memory.py
│   ├── event_visualization.py
│   └── pareto_curve.py
├── baselines/                # Baseline 复现/接口
│   ├── diffusion_policy/
│   ├── flow_matching_policy/
│   ├── consistency_policy/
│   ├── probeflow/
│   └── act/
└── scripts/
    ├── run_all_baselines.sh
    ├── run_ablation.sh
    └── run_visualization.sh
```

### 6.6 每日同步机制

```
每天 15 分钟站会（建议固定时间，如早上 10:00）:
  1. 昨天完成了什么
  2. 今天计划做什么
  3. 当前有没有 blocker

关键决策点（需要三人一起讨论）:
  ├── Week 1 Day 7: 接口对齐会议（30 分钟）
  ├── Week 2 Day 4: Phase 1 训练结果 review（关键！）
  ├── Week 3 Day 3: 实验数字中期 review，确认故事线
  └── Week 4 Day 5: 论文终稿 review

共享工具:
  ├── Git: 统一 repo，每人一个 feature branch，主分支受保护
  ├── W&B: 统一 project，实验命名规范 {method}_{dataset}_{seed}
  └── Overleaf: 论文共同编辑，C 负责最终格式统一
```

---

## 7. 参考文献（核心）

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
2. Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
3. Høeg et al., "Streaming Diffusion Policy", arXiv 2024
4. Fang et al., "ProbeFlow: Training-Free Adaptive Flow Matching for VLA Models", arXiv 2026
5. Jia et al., "A2A: Action-to-Action Flow Matching", arXiv 2026
6. Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)", RSS 2023
7. Song et al., "Consistency Models", ICML 2023
8. Oquab et al., "DINOv2: Learning Robust Visual Features", TMLR 2024
9. Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", NeurIPS 2023