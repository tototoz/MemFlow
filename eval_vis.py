#!/usr/bin/env python3
"""
带 MuJoCo 实时窗口的评估脚本，支持保存为 .mp4 视频文件

用法:
    # 仅显示实时窗口
    python eval_vis.py --model_path /home/ydj/article/LIBERO/experiments/LIBERO_SPATIAL/Multitask/MemFlowPolicy_seed0/run_001/multitask_model_ep30.pth --task_id 1

    # 显示实时窗口并保存视频
    python eval_vis.py --model_path /home/ydj/article/LIBERO/experiments/LIBERO_SPATIAL/Multitask/FlowMatchingPolicy_seed0/run_001/multitask_model_ep25.pth --task_id 1 --save_video

    # 保存视频到指定目录
    python eval_vis.py --model_path experiments/.../model.pth --task_id 1 --save_video --video_dir my_videos
"""

import argparse
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "policy"))

import numpy as np
import torch
import cv2

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.lifelong.algos import Sequential
from libero.lifelong.datasets import get_dataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.utils import torch_load_model, safe_device, control_seed
from libero.lifelong.main import get_task_embs

try:
    from diffusion_policy.diffusion_policy import DiffusionPolicy  # noqa
except Exception as e:
    print(f"Warning: Could not load DiffusionPolicy: {e}")

try:
    from flow_matching.flow_matching_policy import FlowMatchingPolicy  # noqa
except Exception as e:
    print(f"Warning: Could not load FlowMatchingPolicy: {e}")

try:
    from memflow.memflow_policy import MemFlowPolicy  # noqa
except Exception as e:
    print(f"Warning: Could not load MemFlowPolicy: {e}")

try:
    from act.act_policy import ACTPolicy  # noqa
except Exception as e:
    print(f"Warning: Could not load ACTPolicy: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="权重文件路径，例如 experiments/.../task9_model.pth")
    parser.add_argument("--task_id", type=int, default=0,
                        help="评估哪个任务 (0-9)")
    parser.add_argument("--n_eval", type=int, default=5,
                        help="评估次数")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_video", action="store_true",
                        help="保存为 .mp4 文件")
    parser.add_argument("--video_dir", type=str, default="videos",
                        help="视频保存目录")
    parser.add_argument("--video_res", type=int, default=512,
                        help="视频分辨率（正方形），默认 512")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="覆盖模型训练时的 benchmark，例如 LIBERO_10（不填则使用模型内置配置）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载权重和配置
    print(f"[info] 加载模型: {args.model_path}")
    sd, cfg, _ = torch_load_model(args.model_path, map_location=args.device)

    cfg.device = args.device
    cfg.eval.use_mp = False
    cfg.eval.n_eval = args.n_eval

    # 覆盖 benchmark（允许跨 benchmark 评估，例如 LIBERO_90 训练 → LIBERO_10 评估）
    if args.benchmark is not None:
        print(f"[info] 覆盖 benchmark: {cfg.benchmark_name} → {args.benchmark}")
        cfg.benchmark_name = args.benchmark

    # 覆盖旧 checkpoint 中训练机器的路径
    cfg.bert_cache_dir = "/home/ydj/bert"


    # 修复旧配置中训练机器的路径（无条件覆盖，旧 checkpoint 可能含 /robot 路径）
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    control_seed(cfg.seed)

    # 先加载 benchmark，获取 n_tasks
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # 初始化算法和策略（n_tasks 来自 benchmark，兼容不同规模）
    algo = safe_device(Sequential(benchmark.n_tasks, cfg), cfg.device)
    algo.policy.load_state_dict(sd)
    algo.eval()

    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)

    # 初始化 ObsUtils（raw_obs_to_tensor_obs 依赖此初始化）
    # 不需要真实数据集文件，只需用 obs spec 初始化即可
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})

    print(f"\n[info] 任务: {task.language}")
    print(f"[info] 评估 {args.n_eval} 次，使用 MuJoCo 实时窗口\n")

    # 加载初始状态
    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path)

    # 创建带实时窗口的环境
    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h,
        "camera_widths": cfg.data.img_w,
        "has_renderer": True,           # 开启 MuJoCo 实时窗口
        "has_offscreen_renderer": True, # 同时保留 offscreen 用于策略观测
        "render_camera": "agentview",   # 窗口视角
    }
    env = ControlEnv(**env_args)

    num_success = 0

    # 创建视频保存目录
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"[info] 视频将保存到: {args.video_dir}")

    for ep in range(args.n_eval):
        env.reset()
        init_state = init_states[ep % init_states.shape[0]]
        env.env.sim.set_state_from_flattened(np.array(init_state))
        env.env.sim.forward()

        # 预热物理仿真
        dummy = np.zeros(7)
        obs = None
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)
            env.env.render()

        # 初始化视频录制
        video_writer = None
        if args.save_video:
            video_path = os.path.join(args.video_dir, f"task_{args.task_id}_ep_{ep+1}.mp4")
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,  # FPS = control_freq
                (args.video_res, args.video_res)
            )
            print(f"[info] 录制视频: {video_path}")

        algo.reset()
        done = False
        task_success = False
        steps = 0

        while steps < cfg.eval.max_steps and not done:
            steps += 1

            # 策略推理
            data = raw_obs_to_tensor_obs([obs], task_emb, cfg)
            with torch.no_grad():
                action = algo.policy.get_action(data)

            obs, reward, done, info = env.step(action[0])
            task_success = task_success or info.get("success", False)
            env.env.render()  # 刷新实时窗口

            # 保存视频帧：用 sim.render 高分辨率渲染，不受策略输入分辨率限制
            if video_writer is not None:
                frame = env.env.sim.render(
                    height=args.video_res, width=args.video_res, camera_name="agentview"
                )[::-1]                   # MuJoCo 图像上下翻转，需要纠正
                frame_bgr = frame[:, :, ::-1].copy()   # RGB → BGR for cv2
                video_writer.write(frame_bgr)

        # 释放视频写入器
        if video_writer is not None:
            video_writer.release()
            print(f"[info] 视频已保存: {video_path}")

        success = int(task_success)
        num_success += success
        print(f"  Episode {ep+1}/{args.n_eval}: {'成功' if success else '失败'} ({steps} 步)")

    env.env.close()
    success_rate = num_success / args.n_eval
    print(f"\n[result] Task {args.task_id} 成功率: {success_rate:.2%} ({num_success}/{args.n_eval})")

    if args.save_video:
        print(f"\n[info] 所有视频已保存到: {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    main()
