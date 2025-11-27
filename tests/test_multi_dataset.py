#!/usr/bin/env python3

"""
测试多数据集融合功能

使用方法:
python test_multi_dataset.py --data-dir data_expert_augmented_small200k --batch-size 4
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加tools路径
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from aether_lite_dataset import create_dataloaders, AetherLiteDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_loading(data_dir: str, batch_size: int = 4):
    """测试数据集加载功能"""
    logger.info("=== 测试多数据集融合 ===")

    # 自定义数据集占比
    custom_ratios = {
        "burst_inpaint": 0.4,  # 突发修复场景占40%
        "harmonic": 0.3,       # 谐波场景占30%
        "low_snr": 0.2,        # 低信噪比场景占20%
        "transient": 0.1       # 瞬态场景占10%
    }

    try:
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2,
            dataset_ratios=custom_ratios
        )

        # 测试训练数据
        logger.info("\n=== 测试训练数据加载 ===")
        for i, batch in enumerate(train_loader):
            logger.info(f"批次 {i}:")
            logger.info(f"  音频: {batch['audio'].shape}")
            logger.info(f"  特征: {batch['features'].shape}")
            logger.info(f"  StableCodec: {batch['stable_codec'].shape}")
            logger.info(f"  数据集来源: {batch.get('dataset_names', 'N/A')}")
            logger.info(f"  音频长度: {batch['audio_lengths']}")
            logger.info(f"  特征长度: {batch['feature_lengths']}")

            if i >= 2:  # 只测试前3个批次
                break

        # 测试验证数据
        logger.info("\n=== 测试验证数据加载 ===")
        for i, batch in enumerate(val_loader):
            logger.info(f"批次 {i}:")
            logger.info(f"  音频: {batch['audio'].shape}")
            logger.info(f"  特征: {batch['features'].shape}")
            logger.info(f"  StableCodec: {batch['stable_codec'].shape}")

            if i >= 1:  # 只测试前2个批次
                break

        # 统计数据集分布
        logger.info("\n=== 数据集分布统计 ===")
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'get_dataset_info'):
            info = train_dataset.get_dataset_info()
            logger.info(f"总样本数: {info['total_samples']}")
            logger.info("各数据集样本分布:")
            for dataset_name, count in info['samples_per_dataset'].items():
                ratio = count / info['total_samples']
                logger.info(f"  {dataset_name}: {count} 样本 ({ratio:.1%})")

        logger.info("\n=== 测试完成 ===")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_individual_dataset(data_dir: str):
    """测试单个数据集加载"""
    logger.info("=== 测试单个数据集 ===")

    try:
        dataset = AetherLiteDataset(
            data_dir=data_dir,
            split="train",
            dataset_ratios={"burst_inpaint": 1.0}  # 只使用burst_inpaint
        )

        logger.info(f"数据集大小: {len(dataset)}")

        # 测试单个样本
        sample = dataset[0]
        logger.info("第一个样本:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key}: {value.shape} {value.dtype}")
            else:
                logger.info(f"  {key}: {value}")

        logger.info("单个数据集测试完成")

    except Exception as e:
        logger.error(f"单个数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="测试多数据集融合")
    parser.add_argument("--data-dir", default="data_expert_augmented_small200k",
                       help="数据目录路径")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--test-individual", action="store_true",
                       help="测试单个数据集")

    args = parser.parse_args()

    # 检查数据目录
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_path}")
        return

    if args.test_individual:
        test_individual_dataset(args.data_dir)
    else:
        test_dataset_loading(args.data_dir, args.batch_size)

if __name__ == "__main__":
    main()