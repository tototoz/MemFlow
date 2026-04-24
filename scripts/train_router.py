"""
Phase 2: Train Adaptive Denoising Step Router (ADSR).

Usage:
    python scripts/train_router.py \
        --checkpoint experiments/LIBERO_SPATIAL_memflow/model.pth \
        --benchmark LIBERO_SPATIAL \
        --label_tolerance 0.01 \
        --n_epochs 30

This script:
1. Loads a Phase 1 DDPM checkpoint (MemFlowPolicy with router.enabled=false)
2. Enables the router and freezes all other parameters
3. Generates difficulty labels by running DDIM with each K on training data
4. Trains the router with CE + cost loss
5. Saves the updated checkpoint
"""

import os
import sys
import argparse
import pprint
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libero.lifelong.datasets import get_dataset, SequenceVLDataset
from libero.lifelong.utils import get_task_embs, control_seed
from libero.libero.benchmark import get_benchmark


def generate_difficulty_labels(
    policy, dataloader, step_choices, label_tolerance, n_eval_seeds=4, device="cuda"
):
    """
    Generate ground-truth difficulty labels for router training.

    For each training sample:
    1. Get ground-truth action target from dataset
    2. Run DDIM with each K in step_choices, compute MSE vs ground truth
    3. Assign label = index of smallest K with MSE < tolerance

    Returns:
        embeddings: (N, cond_dim) observation condition vectors
        labels: (N,) integer labels
        errors: (N, len(step_choices)) MSE values for analysis
    """
    policy.eval()
    all_embeddings = []
    all_errors = []

    n_obs_steps = policy.n_obs_steps
    horizon = policy.horizon

    print(f"\n[Router] Generating difficulty labels...")
    print(f"  step_choices: {step_choices}")
    print(f"  label_tolerance: {label_tolerance}")
    print(f"  n_eval_seeds: {n_eval_seeds}")

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Labeling")):
            data = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}
            if "obs" in data:
                data["obs"] = {
                    k: v.to(device) for k, v in data["obs"].items()
                }

            data = policy.preprocess_input(data, train_mode=False)
            obs_cond = policy.encode_obs(data)
            all_embeddings.append(obs_cond.cpu())

            start_idx = n_obs_steps - 1
            gt_action = data["actions"][:, start_idx:start_idx + horizon, :]
            B = gt_action.shape[0]

            batch_errors = torch.zeros(B, len(step_choices))
            for j, K in enumerate(step_choices):
                mse_sum = torch.zeros(B, device=device)
                for seed in range(n_eval_seeds):
                    torch.manual_seed(batch_idx * 1000 + seed)
                    pred_action = policy._ddim_sample(obs_cond, K)
                    mse = ((pred_action - gt_action) ** 2).mean(dim=(1, 2))
                    mse_sum += mse
                batch_errors[:, j] = (mse_sum / n_eval_seeds).cpu()

            all_errors.append(batch_errors)

    embeddings = torch.cat(all_embeddings, dim=0)
    errors = torch.cat(all_errors, dim=0)

    labels = torch.full((embeddings.shape[0],), len(step_choices) - 1, dtype=torch.long)
    for i in range(len(step_choices) - 1):
        mask = errors[:, i] < label_tolerance
        labels[mask] = torch.minimum(labels[mask], torch.tensor(i))

    label_dist = {
        step_choices[i]: (labels == i).sum().item()
        for i in range(len(step_choices))
    }
    print(f"\n[Router] Label distribution: {label_dist}")
    print(f"  Total samples: {embeddings.shape[0]}")
    print(f"  Mean error per K: {errors.mean(dim=0).tolist()}")

    return embeddings, labels, errors


def train_router(policy, embeddings, labels, cfg):
    """Train the step router with frozen UNet."""
    device = next(policy.parameters()).device

    for param in policy.parameters():
        param.requires_grad = False
    for param in policy.step_router.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in policy.step_router.parameters())
    total = sum(p.numel() for p in policy.parameters())
    print(f"\n[Router] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=RandomSampler(dataset),
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        policy.step_router.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"], eta_min=1e-6
    )

    step_choices_t = torch.tensor(
        policy.step_choices, device=device, dtype=torch.float32
    )
    K_max = float(max(policy.step_choices))
    cost_weight = policy.router_cost_weight

    print(f"\n[Router] Training for {cfg['n_epochs']} epochs...")
    policy.step_router.train()

    for epoch in range(cfg["n_epochs"]):
        total_loss = 0.0
        total_ce = 0.0
        total_cost = 0.0
        n_batches = 0

        for emb_batch, label_batch in dataloader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            step_weights, logits = policy.step_router(emb_batch, training=True)
            ce_loss = F.cross_entropy(logits, label_batch, label_smoothing=0.1)
            cost_loss = (step_weights * step_choices_t).sum(dim=-1).mean() / K_max
            loss = ce_loss + cost_weight * cost_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_cost += cost_loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            policy.step_router.eval()
            with torch.no_grad():
                all_emb = embeddings.to(device)
                _, logits = policy.step_router(all_emb, training=False)
                preds = logits.argmax(dim=-1).cpu()
                acc = (preds == labels).float().mean().item()
                dist = {
                    policy.step_choices[i]: (preds == i).sum().item()
                    for i in range(len(policy.step_choices))
                }
                mean_K = step_choices_t[preds.to(device)].mean().item()
            policy.step_router.train()

            print(
                f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | "
                f"loss={total_loss/n_batches:.4f} | "
                f"ce={total_ce/n_batches:.4f} | "
                f"cost={total_cost/n_batches:.4f} | "
                f"acc={acc:.3f} | "
                f"mean_K={mean_K:.1f} | "
                f"dist={dist}"
            )

    policy.step_router.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train ADSR Router")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Phase 1 checkpoint (.pth)")
    parser.add_argument("--benchmark", type=str, default="LIBERO_SPATIAL")
    parser.add_argument("--task_order_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--label_tolerance", type=float, default=0.01)
    parser.add_argument("--n_eval_seeds", type=int, default=4)
    parser.add_argument("--label_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=None,
                        help="Output checkpoint path (default: same dir as input)")
    args = parser.parse_args()

    control_seed(args.seed)

    print("="*70)
    print("Phase 2: Adaptive Denoising Step Router Training")
    print("="*70)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))

    # ---- Load checkpoint ----
    print(f"\n[Router] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
    else:
        raise ValueError("Checkpoint must contain 'cfg' key")

    if "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError("Checkpoint must contain 'policy_state_dict' or 'state_dict'")

    shape_meta = ckpt.get("shape_meta", cfg.get("shape_meta", None))
    if shape_meta is None:
        raise ValueError("Checkpoint must contain 'shape_meta'")

    # Enable router in config
    if not hasattr(cfg.policy, "router"):
        from easydict import EasyDict
        cfg.policy.router = EasyDict({
            "enabled": True,
            "step_choices": [10, 25, 50, 100],
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "cost_weight": 0.1,
            "temperature": 1.0,
        })
    else:
        cfg.policy.router.enabled = True

    # Import and construct policy
    from memflow.memflow_policy import MemFlowPolicy
    policy = MemFlowPolicy(cfg, shape_meta)

    # Load Phase 1 weights (router weights will be randomly initialized)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    print(f"  Loaded state_dict: {len(state_dict)} keys")
    if missing:
        print(f"  Missing (expected for router): {missing}")
    if unexpected:
        print(f"  Unexpected: {unexpected}")

    policy = policy.to(args.device)

    # ---- Build DDIM subsequences (already done in __init__ if router enabled) ----

    # ---- Load training data ----
    print(f"\n[Router] Loading benchmark: {args.benchmark}")
    benchmark = get_benchmark(args.benchmark)(args.task_order_index)

    from libero.libero import get_libero_path
    folder = cfg.get("folder", None) or get_libero_path("datasets")

    datasets = []
    descriptions = []
    for i in range(benchmark.n_tasks):
        ds, sm = get_dataset(
            dataset_path=os.path.join(folder, benchmark.get_task_demonstration(i)),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=(i == 0),
            seq_len=cfg.data.seq_len,
        )
        descriptions.append(benchmark.get_task(i).language)
        task_embs = get_task_embs(cfg, descriptions)
        datasets.append(SequenceVLDataset(ds, task_embs[i]))

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset(datasets)
    label_loader = DataLoader(
        combined,
        batch_size=args.label_batch_size,
        shuffle=False,
        num_workers=2,
    )

    # ---- Generate labels ----
    embeddings, labels, errors = generate_difficulty_labels(
        policy, label_loader, policy.step_choices,
        args.label_tolerance, args.n_eval_seeds, args.device,
    )

    # ---- Train router ----
    train_cfg = {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    policy = train_router(policy, embeddings, labels, train_cfg)

    # ---- Save checkpoint ----
    output_path = args.output
    if output_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(ckpt_dir, "model_with_router.pth")

    save_dict = {
        "cfg": cfg,
        "shape_meta": shape_meta,
        "policy_state_dict": policy.state_dict(),
        "router_labels": {
            "tolerance": args.label_tolerance,
            "errors_mean": errors.mean(dim=0).tolist(),
            "label_distribution": {
                policy.step_choices[i]: (labels == i).sum().item()
                for i in range(len(policy.step_choices))
            },
        },
    }
    torch.save(save_dict, output_path)
    print(f"\n[Router] Saved checkpoint: {output_path}")
    print("="*70)
    print("Phase 2 complete!")
    print("="*70)


if __name__ == "__main__":
    main()
