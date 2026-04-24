# MemFlow 论文数学证明与实验设计

## Part 1: 数学证明

---

### Theorem 1: 收敛速度与条件分布复杂度

**Theorem 1.** 设 DDPM 前向过程由线性调度 β_s ∈ [1e-4, 0.02] 定义，ᾱ_t = ∏(1-β_s)。DDIM 第 i 步的 Tweedie denoised estimate 为：

```
â₀^{(i)} = (x_{τᵢ} - √(1-ᾱ_{τᵢ}) · ε_θ(x_{τᵢ}, τᵢ, c)) / √(ᾱ_{τᵢ})
```

则：

**(Case 1) 单峰高斯：** 若 p(a₀|c) = N(μ_c, σ²_c I)，且 ε_θ 完美学习，则

```
Δ^{(i)} = ||â₀^{(i)} - â₀^{(i-1)}||² / (H·dₐ) = O(1/SNR(τᵢ)²)
```

其中 SNR(t) = ᾱ_t/(1-ᾱ_t)。Δ^{(i)} 单调递减，按 SNR 指数衰减。

**(Case 2) K-模混合高斯：** 若 p(a₀|c) = Σ w_k N(μ_k, σ²_k I)，最小模间距 d_min = min_{j≠k}||μ_j-μ_k||，则收敛分两阶段：

- **Phase I（mode selection）：** 当 SNR(τᵢ) < d²_min/(4σ²_max) 时，score 函数 Lipschitz 常数 L_s = Θ(K·d_min/σ²_max)，Δ^{(i)} = Ω(d²_min/(H·dₐ))
- **Phase II（intra-mode refinement）：** 当 SNR(τᵢ) > d²_min/(4σ²_max) 时，退化为 Case 1

**Proof (Case 1).**

对 p(a₀|c) = N(μ_c, σ²_c I)，时间 t 的后验为 p(a₀|x_t, c) = N(μ̃_t, σ̃²_t I)。Tweedie 估计为后验均值：

```
â₀^{(i)} = E[a₀|x_{τᵢ}] = μ_c + γᵢ · (a₀ - μ_c + ε/√SNR(τᵢ))
```

其中 γᵢ = SNR(τᵢ)·σ²_c / (1 + SNR(τᵢ)·σ²_c)。

由于 DDIM 是确定性的（同一个 ε 驱动整条轨迹），连续两步的差只取决于系数变化：

```
||â₀^{(i)} - â₀^{(i-1)}||² ∝ |γᵢ - γᵢ₋₁|² · ||a₀ - μ_c + ε/√SNR||²
```

而 dγ/d(SNR) = σ²_c/(1+SNR·σ²_c)²，在线性调度下 SNR 随 t→0 近似指数增长，故 Δ^{(i)} 指数衰减。□

**Proof (Case 2).**

混合分布的 score 为：

```
∇ log p(x_t|c) = -1/(1-ᾱ_t) · Σ_k r_k(x_t,t) · (x_t - √ᾱ_t·μ_k)
```

其中 r_k 是后验混合权重。当 SNR 低（噪声大）时，r_k ≈ 均匀，Tweedie 估计是所有 μ_k 的加权平均。SNR 增大时权重锐化，转变点约在 √ᾱ_t·d_min/√(1-ᾱ_t) ≈ 2，即 SNR ≈ d²_min/(4σ²_max)。

转变期间 Tweedie 估计在模之间跳跃，Δ^{(i)} = Ω(d²_min/(H·dₐ))，score 的 Lipschitz 常数下界为 Θ(K·d_min/σ²_max)。

模选择后，后验集中于单模 k*，退化为 Case 1。□

---

### Theorem 2: 截断误差上界

**Theorem 2.** 设 â₀^{(1)}, ..., â₀^{(K)} 为 K 步 DDIM 的 Tweedie 估计序列。若 monitor 在步 i* 正确识别 ε-收敛（即 Δ^{(j)} ≤ ε，∀j ≥ i*），则：

```
||â₀^{(K)} - â₀^{(i*)}||² / (H·dₐ) ≤ (K-i*)² · ε
```

**Proof.** 由 telescoping sum：

```
â₀^{(K)} - â₀^{(i*)} = Σ_{j=i*+1}^{K} (â₀^{(j)} - â₀^{(j-1)})
```

三角不等式：

```
||â₀^{(K)} - â₀^{(i*)}|| ≤ Σ_{j=i*+1}^{K} ||â₀^{(j)} - â₀^{(j-1)}||
```

Cauchy-Schwarz：

```
(Σ ||·||)² ≤ (K-i*) · Σ ||·||²
```

代入 Δ^{(j)} ≤ ε：

```
Σ ||â₀^{(j)} - â₀^{(j-1)}||² ≤ (K-i*) · H·dₐ · ε
```

合并：

```
||â₀^{(K)} - â₀^{(i*)}||² ≤ (K-i*)² · H·dₐ · ε
```

两边除以 H·dₐ 即得。□

---

### Corollary 1: 几何衰减界

**Corollary 1.** 若步 i* 之后 Δ 几何衰减：Δ^{(j)} ≤ Δ^{(i*)} · ρ^{j-i*}（0 < ρ < 1），则：

```
||â₀^{(K)} - â₀^{(i*)}||² ≤ H·dₐ · Δ^{(i*)} / (1-ρ)²
```

**此界与 K 无关**。

**Proof.** 由三角不等式展开并求和几何级数：

```
||...|| ≤ √(H·dₐ·Δ^{(i*)}) · Σ_{m=1}^{K-i*} ρ^{m/2} ≤ √(H·dₐ·Δ^{(i*)}) · √ρ/(1-√ρ)
```

平方后利用 √ρ/(1-√ρ) ≤ 1/(1-ρ)：

```
||...||² ≤ H·dₐ · Δ^{(i*)} / (1-ρ)²   □
```

---

### Proposition 1: 最优停止准则

**Proposition 1.** 考虑质量-效率权衡：

```
J(i) = E[||â₀^{(K)} - â₀^{(i)}||²/(H·dₐ)] + μ·i
```

其中 μ > 0 是每步计算代价。最优停止步 i* 满足：

```
Δ^{(i*)} ≈ μ / (K-i*)
```

**解释：** 当多走一步的边际误差减少量（≈(K-i)·Δ^{(i)}）等于边际代价 μ 时停止。Monitor 的阈值 θ 是这个条件的学习代理。

---

### Proposition 2: Monitor 泛化性

**Proposition 2.** 在相同噪声调度下训练的 monitor 可跨任务分布迁移，因为：

1. **调度决定衰减形状：** Δ 的衰减由 ᾱ_τᵢ 调度和 score 平滑性决定，前者跨任务相同
2. **归一化特征：** 12 维特征包含 Δ/logΔ/Δ ratio（无标度收敛描述符）、progress/SNR（调度相对量），不依赖具体动作值
3. **任务无关的收敛模式：** 不同任务只是 Phase I→II 转变点不同（由分布复杂度决定），但 Δ 的衰减曲线形状定性相似
4. **充分条件：** 若两组任务的 score Lipschitz 常数 L_s^{(1)}/L_s^{(2)} = O(1)，则 Δ 衰减曲线只差有界缩放，monitor 的 log 和 ratio 特征对此不变

---

## Part 2: 完整实验设计

### 实验总览

| # | 实验 | 载体 | 目的 |
|---|------|------|------|
| 1 | 主结果 | Table 1 | DCM vs 基线的 success rate & steps |
| 2 | 动机图 | Figure 1 | 不同任务的 Δ 衰减曲线差异 |
| 3 | DCM 可视化 | Figure 2 | 单条轨迹的 â₀/Δ/h 演化 |
| 4 | Pareto 曲线 | Figure 3 | DCM 主导效率-质量前沿 |
| 5 | 步数分布 | Figure 4 | 不同任务的步数直方图 |
| 6 | 消融 | Table 2 | 各组件贡献 |
| 7 | 跨 suite 泛化 | Table 3 | 验证 Proposition 2 |
| 8 | 误差上界验证 | Figure 5 | 验证 Theorem 2 |

---

### Experiment 1: 主结果 (Table 1)

**方法：**

| Method | 描述 | 推理方式 |
|--------|------|---------|
| DDPM-100 | 标准 DiffusionPolicy | 100 步 DDPM |
| DDIM-50 | 同一 UNet，DDIM 推理 | 50 步 DDIM |
| DDIM-25 | 同上 | 25 步 |
| DDIM-10 | 同上 | 10 步 |
| DCM-0.5 | MemFlowPolicy, dcm=true | 自适应 (threshold=0.5) |
| DCM-0.7 | 同上 | 自适应 (threshold=0.7) |
| DCM-0.9 | 同上 | 自适应 (threshold=0.9) |

**Benchmark：** LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL

**Seeds：** 3 (42, 123, 456)

**命令：**
```bash
# DDPM-100 baseline
python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL \
  seed=42 lifelong=multitask train.n_epochs=50

# DCM 端到端
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
  seed=42 lifelong=multitask train.n_epochs=50 policy.dcm.enabled=true

# DDIM-K: 复用 DDPM-100 的权重，评估时改推理步数
```

**指标：** success rate (% ± std), mean steps, wall-clock time/action

**预期：** DCM-0.7 达到 DDIM-50 的 success rate，平均仅用 ~15-25 步

**GPU 时间：** 3 suites × 2 训练配置 × 3 seeds × 6h = ~108h

---

### Experiment 2: 动机图 (Figure 1)

**目的：** 论文第一张图，展示"不同任务收敛速度差异大"这个动机

**方法：** 取训好的 DDPM 模型，对 4 个代表性任务各跑 100 次 DDIM-50（无 early stop），记录完整 Δ^{(i)} 轨迹

**选任务（从 LIBERO_SPATIAL）：**
- 简单 pick-and-place（快收敛）
- 多步操作（慢收敛）
- 精密放置（中等）
- 多策略任务（多峰）

**画法：** 每任务一条曲线，x=step, y=Δ (log scale)，mean + shaded std

**GPU 时间：** ~2h（推理）

---

### Experiment 3: DCM 可视化 (Figure 2)

**4 面板：**
- A: â₀ 的前 2 个动作维度随步数变化
- B: Δ^{(i)} (log scale) + ε 阈值线
- C: Monitor 输出 h^{(i)} + threshold θ 线，标注停止点
- D: 机器人执行该动作的效果

**GPU 时间：** <1h

---

### Experiment 4: Pareto 曲线 (Figure 3)

**数据：** Experiment 1 结果 + 额外 DCM threshold sweep

**DCM thresholds：** 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95

**DDIM baselines：** K = 10, 15, 20, 25, 30, 40, 50

所有 DCM 点共享同一模型（threshold 仅影响推理），所有 DDIM 点共享同一 UNet

**预期：** DCM 曲线严格在 DDIM-K 曲线上方（相同步数更高 success rate）

**GPU 时间：** ~12h（推理）

---

### Experiment 5: 步数分布 (Figure 4)

**方法：** DCM (threshold=0.7) 在 LIBERO_SPATIAL 全 10 个任务上推理，100 rollouts/task，记录 steps_used

**画法：** 按任务着色的直方图 or violin plot

**预期：** 简单任务聚在 8-12 步，复杂任务 20-35 步

**GPU 时间：** ~2h

---

### Experiment 6: 消融 (Table 2)

| 消融维度 | 变体 | Config |
|---------|------|--------|
| Monitor 类型 | DCM-CA (full) | `use_cond=true` (默认) |
| | DCM-Base | `use_cond=false` |
| K_train | 10, **20**, 30 | `K_train=10/20/30` |
| lambda_dcm | 0.01, **0.1**, 0.5 | `lambda_dcm=...` |
| threshold | 0.5, **0.7**, 0.9 | 推理时改 |
| 训练方式 | **端到端** | 默认 |
| | 两阶段 | 先 50 epoch DDPM，再冻结训 DCM |

Benchmark: LIBERO_SPATIAL, seed=42 (确认 top 2-3 用 3 seeds)

**GPU 时间：** ~108h

---

### Experiment 7: 跨 Suite 泛化 (Table 3)

| 训练 Suite | 测试 Suite | DCM 来源 |
|-----------|-----------|---------|
| SPATIAL | SPATIAL | 自身 |
| SPATIAL | OBJECT | 迁移 |
| SPATIAL | GOAL | 迁移 |
| OBJECT | OBJECT | 自身 |
| GOAL | GOAL | 自身 |

**方法：** 迁移时取测试 suite 的 UNet（dcm=false 训练），挂上训练 suite 的 DCM monitor 权重

**Seeds：** 3

**GPU 时间：** ~108h（部分与 Exp 1 重叠）

---

### Experiment 8: 误差上界验证 (Figure 5)

**方法：** DCM 推理时同时记录：
- â₀^{(i*)}（停止时）
- â₀^{(K)}（跑满 K_max 步，不管 DCM 判断）

画 scatter plot：x = 理论上界 (K-i*)²·Δ^{(i*)}，y = 实际误差

**预期：** 所有点在对角线下方，大部分点远低于上界（保守但有效）

**GPU 时间：** ~4h

---

### GPU 预算总结

| 实验 | GPU-hours |
|------|-----------|
| Exp 1 主结果 | 108 |
| Exp 2 动机图 | 2 |
| Exp 3 可视化 | 1 |
| Exp 4 Pareto | 12 |
| Exp 5 分布 | 2 |
| Exp 6 消融 | 108 |
| Exp 7 泛化 | 108 |
| Exp 8 误差 | 4 |
| **合计** | **~345 A100-hours ≈ 15 A100-days** |

（模型复用后实际更少，约 250h）

---

### 论文结构建议

**Main text (6 pages)：**
- Introduction (1p): 动机（Figure 1）+ 贡献总结
- Background (0.5p): DDPM, DDIM, Diffusion Policy
- Method (1.5p): DCM 设计 + Theorem 2 + Corollary 1 + 端到端训练
- Experiments (2.5p): Table 1 + Figure 1-4 + Table 2
- Conclusion (0.5p)

**Appendix：**
- Theorem 1 完整证明 (Case 1 + Case 2)
- Proposition 1-2
- Table 3（泛化）+ Figure 5（误差验证）
- 超参细节 + 额外可视化
