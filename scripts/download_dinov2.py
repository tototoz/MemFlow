#!/usr/bin/env python3
"""
预下载 DINOv2 模型到本地目录。

用法:
    python scripts/download_dinov2.py                    # 默认下载 dinov2-small
    python scripts/download_dinov2.py --model dinov2-base  # 下载 base 版本
    python scripts/download_dinov2.py --output_dir /path/to/models  # 指定输出目录
"""

import argparse
import os
from pathlib import Path


def download_dinov2(model_name: str, output_dir: str):
    """下载 DINOv2 模型到指定目录"""
    print(f"[download] 开始下载 {model_name}...")

    from transformers import AutoModel

    # DINOv2 在 Hugging Face 上用连字符
    hf_model_name = model_name.replace("_", "-")

    # 下载模型
    model = AutoModel.from_pretrained(f"facebook/{hf_model_name}")

    # 保存到本地（保存时也用连字符名称）
    save_path = os.path.join(output_dir, hf_model_name)
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)

    print(f"[download] 模型已保存到: {save_path}")
    print(f"[download] 文件大小: {sum(f.stat().st_size for f in Path(save_path).glob('**/*') if f.is_file()) / 1024 / 1024:.1f} MB")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="预下载 DINOv2 模型")
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2-small",
        choices=["dinov2-small", "dinov2-base", "dinov2-large", "dinov2_small", "dinov2_base"],
        help="DINOv2 模型版本 (推荐用连字符格式)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (默认: LIBERO/dinov2_models)"
    )
    args = parser.parse_args()

    # 统一转换为连字符格式
    model_name = args.model.replace("_", "-")

    # 默认保存到项目目录下
    if args.output_dir is None:
        project_root = Path(__file__).parent.parent
        args.output_dir = project_root / "dinov2_models"

    save_path = download_dinov2(model_name, str(args.output_dir))

    print("\n[done] 下载完成！使用方法:")
    print(f"  方式1 - 命令行:")
    print(f"    python train.py policy=memflow_policy policy.dinov2_local_path={save_path}")
    print(f"  方式2 - 配置文件 (memflow_policy.yaml):")
    print(f"    dinov2_local_path: {save_path}")


if __name__ == "__main__":
    main()
