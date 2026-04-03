#!/usr/bin/env python3
"""
Training script for LIBERO benchmark with different policies
Usage:
    python train.py policy=bc_transformer_policy benchmark_name=LIBERO_SPATIAL seed=42
    python train.py policy=diffusion_policy benchmark_name=LIBERO_SPATIAL seed=42
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import sys
import json
import multiprocessing
import pprint
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
import torch.nn as nn
from easydict import EasyDict
from omegaconf import OmegaConf

# Add policy directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'policy'))

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    control_seed,
    safe_device,
    create_experiment_dir,
    get_task_embs,
)

# Register DiffusionPolicy AFTER libero is fully loaded (avoids circular import)
try:
    from diffusion_policy.diffusion_policy import DiffusionPolicy  # noqa: F401
except Exception as e:
    print(f"Warning: Could not load DiffusionPolicy: {e}")


def print_available_policies():
    """Print all available policies"""
    print("\n" + "="*70)
    print("Available Policies:")
    print("="*70)
    for name in sorted(get_policy_list().keys()):
        print(f"  - {name}")
    print("="*70 + "\n")


@hydra.main(config_path="libero/configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # Convert to EasyDict
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    num_gpus = cfg.get("num_gpus", 1)
    available = torch.cuda.device_count()
    if num_gpus > 1:
        assert num_gpus <= available, (
            f"请求 {num_gpus} 卡，但 CUDA_VISIBLE_DEVICES 只暴露了 {available} 张\n"
            f"请在运行前设置: export CUDA_VISIBLE_DEVICES=0,1,2,3"
        )
    cfg.device = cfg.get("device", "cuda")

    # Print available policies
    print_available_policies()

    # Print config
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    # Control seed
    control_seed(cfg.seed)

    # Prepare paths
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    # Load benchmark
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # Load datasets
    print("\n" + "="*70)
    print("Loading Datasets...")
    print("="*70)

    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
            task_description = benchmark.get_task(i).language
            descriptions.append(task_description)
            manip_datasets.append(task_i_dataset)

            print(f"  Task {i+1}/{n_manip_tasks}: {task_description}")
            print(f"    - Demos: {task_i_dataset.n_demos}")
            print(f"    - Sequences: {task_i_dataset.total_num_sequences}")
        except Exception as e:
            print(f"[error] Failed to load task {i}: {benchmark.get_task_names()[i]}")
            print(f"[error] {e}")
            raise

    print("="*70 + "\n")

    # Compute task embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # Create datasets
    gsz = cfg.data.task_group_size
    if gsz == 1:
        datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
        ]
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:
        assert n_manip_tasks % gsz == 0, f"task_group_size does not divide n_tasks"
        datasets = []
        n_demos = []
        n_sequences = []
        for i in range(0, n_manip_tasks, gsz):
            dataset = GroupedTaskDataset(
                manip_datasets[i : i + gsz], task_embs[i : i + gsz]
            )
            datasets.append(dataset)
            n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
            n_sequences.extend(
                [x.total_num_sequences for x in dataset.sequence_datasets]
            )

    n_tasks = n_manip_tasks // gsz

    # Print benchmark info
    print("\n" + "="*70)
    print("Lifelong Benchmark Information")
    print("="*70)
    print(f"Name: {benchmark.name}")
    print(f"# Tasks: {n_tasks}")
    for i in range(n_tasks):
        print(f"  Task {i+1}:")
        for j in range(gsz):
            print(f"    - {benchmark.get_task(i*gsz+j).language}")
    print(f"# demonstrations: {' '.join(f'({x})' for x in n_demos)}")
    print(f"# sequences: {' '.join(f'({x})' for x in n_sequences)}")
    print("="*70 + "\n")

    # Create experiment directory
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name

    # Initialize results
    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),
        "L_fwd": np.zeros((n_manip_tasks,)),
        "S_fwd": np.zeros((n_manip_tasks,)),
    }

    # Get algorithm
    algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)

    # Wrap policy with DataParallel for multi-GPU training
    num_gpus = cfg.get("num_gpus", 1)
    if num_gpus > 1 and torch.cuda.device_count() > 1:
        algo.policy = nn.DataParallel(algo.policy)
        algo.policy.to(cfg.device)
        print(f"[info] DataParallel: 使用 {torch.cuda.device_count()} 张 GPU")

    # Training loop
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)

    if cfg.lifelong.algo == "Multitask":
        # 所有任务数据混合，统一训练一次
        print("\n[info] Multitask 模式：合并所有任务数据统一训练")
        algo.learn_all_tasks(datasets, benchmark, result_summary)
    else:
        # Sequential / ER / EWC 等：逐任务顺序训练
        for task_id in range(n_tasks):
            print(f"\n{'='*70}")
            print(f"Learning Task {task_id+1}/{n_tasks}")
            print(f"{'='*70}")

            algo.learn_one_task(datasets[task_id], task_id, benchmark, result_summary)

            if cfg.eval.eval:
                print(f"\nEvaluating after task {task_id+1}...")
                for prev_task in range(task_id + 1):
                    success_rate = evaluate_success(
                        algo.policy,
                        benchmark,
                        prev_task,
                        cfg,
                        result_summary,
                        task_id,
                    )
                    print(f"  Task {prev_task+1}: {success_rate:.2%} success")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    # Final evaluation
    print("\nFinal Success Rate Matrix:")
    print(result_summary["S_conf_mat"])

    # Save results
    result_file = os.path.join(cfg.experiment_dir, "results.json")
    with open(result_file, "w") as f:
        json.dump(result_summary, f, cls=NpEncoder, indent=2)
    print(f"\nResults saved to: {result_file}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
