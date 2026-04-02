"""训练入口脚本。

使用默认配置在单个 MA-MuJoCo 任务上训练世界模型。
"""
import argparse
import os

import torch

from src.runner.trainer import Trainer


def main() -> None:
    """解析参数并启动训练。"""
    parser = argparse.ArgumentParser(description="MT-MARL 阶段 0 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/phase0",
        help="日志和模型保存目录",
    )
    args = parser.parse_args()

    print(f"设备: {args.device}")
    print(f"配置: {args.config}")
    print(f"输出目录: {args.run_dir}")

    trainer = Trainer(
        config_path=args.config,
        device=args.device,
        run_dir=args.run_dir,
    )
    trainer.run()


if __name__ == "__main__":
    main()
