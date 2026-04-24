# MemFlow: Diffusion Policy with Denoising Convergence Monitor (DCM)

标准 Diffusion Policy 推理固定 100 步去噪。MemFlow 引入 **Denoising Convergence Monitor (DCM)**：在 DDIM 去噪循环中实时监控 Tweedie denoised estimate 的收敛，收敛即停。

## 核心设计

### 端到端联合训练（单阶段）

```
compute_loss(data):
    ddpm_loss = MSE(noise_pred, noise)       # UNet 学去噪
    
    if dcm_enabled:
        with no_grad:                         # UNet 不受 DCM loss 影响
            a0_trajectory = DDIM(K_train步)   # 在线跑 DDIM，记录 â₀ 轨迹
        soft_labels = online_labels(a0_traj)  # 从轨迹生成收敛标签
        dcm_loss = BCE(monitor(features), labels)  # monitor 有梯度
        
    return ddpm_loss + lambda * dcm_loss
```

- UNet 梯度只来自 DDPM loss
- DCM 梯度只来自 DCM loss  
- 同一个 optimizer 更新，互不干扰
- **DCM 标签在线生成**：从当前 UNet 的 â₀ 轨迹计算，随 UNet 共同演化

### 为什么不分阶段？

分阶段（先训 UNet，再冻结训 DCM）意味着 DCM 只见过固定 UNet 的轨迹，换任务分布会失效。端到端训练让 DCM 始终适配当前 UNet 的去噪行为，泛化性更强。

### 推理：在线 Early Stopping

```
for step in DDIM(K_max步):
    eps = UNet(aₜ, t, c)
    â₀ = tweedie_denoise(aₜ, eps, t)
    
    if steps_done >= min_steps:
        h = Monitor(features(â₀, â₀_prev, ...), obs_cond)
        if h > threshold: return â₀    # EARLY STOP
    
    aₜ = ddim_update(â₀, eps, t_prev)
```

## 数学基础

**收敛量**：`Δ = ||â₀^{(i)} - â₀^{(i-1)}||² / (H·dₐ)`

**截断误差上界**：若 Δ < ε 对所有后续步，则 `||â₀^{(K)} - â₀^{(i*)}||² / (H·dₐ) ≤ (K-i*)²·ε`

**收敛速度与分布复杂度**：单峰 → 快收敛（早停），多峰 → 慢收敛（多步）

## 使用

```bash
# 不开 DCM（等同 DiffusionPolicy）
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=100

# 开 DCM（端到端联合训练）
python train.py policy=memflow_policy benchmark_name=LIBERO_SPATIAL \
  lifelong=multitask seed=0 'eval.eval=false' train.n_epochs=100 \
  policy.dcm.enabled=true
```

一条命令，不分阶段。

## 配置

```yaml
dcm:
  enabled: false          # 关闭=纯 DiffusionPolicy
  K_max: 50               # 推理最大 DDIM 步数
  K_train: 20             # 训练时在线 DDIM 步数
  min_steps: 5            # 推理至少跑几步
  threshold: 0.7          # 收敛阈值
  epsilon: 0.001          # 软标签生成阈值
  lambda_dcm: 0.1         # DCM loss 权重
  use_cond: true          # Monitor 是否使用 obs_cond
  hidden_dims: [64, 32]   # Monitor 隐层
```

## 消融

| 实验 | 命令 |
|------|------|
| Baseline (DDPM 100步) | `policy=diffusion_policy` |
| 固定 DDIM-K | `policy=memflow_policy` + 手动改 K_max |
| DCM 完整方法 | `policy=memflow_policy policy.dcm.enabled=true` |
| DCM 不用 obs_cond | 同上 + `policy.dcm.use_cond=false` |

## 参考文献

- **Diffusion Policy**: Chi et al., RSS 2023
- **DDIM**: Song et al., ICLR 2021
- **LIBERO**: Liu et al., NeurIPS 2023
