#!/usr/bin/env python3

"""
Aether-Lite PyTorch数据集类
用于训练和评估的数据加载器

使用方法:
from tools.aether_lite_dataset import AetherLiteDataset, create_dataloaders

# 创建数据集
dataset = AetherLiteDataset("data_aether_lite", split="train")

# 创建数据加载器
train_loader, val_loader = create_dataloaders(
    data_dir="data_aether_lite",
    batch_size=8,
    num_workers=4
)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AetherLiteDataset(Dataset):
    """Aether-Lite训练数据集 - 支持多数据集融合"""

    def __init__(self, data_dir: str, split: str = "train",
                 transform: Optional[callable] = None,
                 dataset_ratios: Optional[Dict[str, float]] = None):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据分割 ("train", "val", "test")
            transform: 数据增强变换函数
            dataset_ratios: 各数据集的占比，例如 {"burst_inpaint": 0.3, "harmonic": 0.3, "low_snr": 0.2, "transient": 0.2}
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # 默认数据集占比（均匀分布）
        self.dataset_ratios = dataset_ratios or {
            "burst_inpaint": 0.25,
            "harmonic": 0.25,
            "low_snr": 0.25,
            "transient": 0.25
        }

        # 验证占比总和
        total_ratio = sum(self.dataset_ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"数据集占比总和为 {total_ratio:.3f}，已自动归一化")
            for key in self.dataset_ratios:
                self.dataset_ratios[key] /= total_ratio

        # 配置信息
        self.config = {
            'sample_rate': 16000,
            'feature_dim': 36,
            'frame_rate': 100,
            'stable_codec_dim': 6
        }

        # 加载多数据集
        self._load_multi_datasets()

    def _load_multi_datasets(self):
        """加载并融合多个数据集"""
        import torch

        # 可用的数据集类型
        dataset_types = ["burst_inpaint", "harmonic", "low_snr", "transient"]

        # 检查数据文件是否存在
        available_datasets = []
        for dataset_name in dataset_types:
            pcm_file = self.data_dir / f"{dataset_name}_200k.pcm"
            f32_file = self.data_dir / f"{dataset_name}_200k_36.f32"
            pt_file = self.data_dir / f"{dataset_name}_200k.pt"

            if all(f.exists() for f in [pcm_file, f32_file, pt_file]):
                available_datasets.append(dataset_name)
            else:
                logger.warning(f"数据集 {dataset_name} 文件不完整，跳过")

        if not available_datasets:
            raise ValueError(f"在 {self.data_dir} 中未找到有效的数据集文件")

        logger.info(f"找到有效数据集: {available_datasets}")

        # 加载各个数据集
        self.datasets = {}
        for dataset_name in available_datasets:
            logger.info(f"加载数据集: {dataset_name}")

            # 加载PCM音频
            pcm_file = self.data_dir / f"{dataset_name}_200k.pcm"
            audio_data = np.frombuffer(open(pcm_file, 'rb').read(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0  # 归一化到[-1,1]

            # 加载F32特征
            f32_file = self.data_dir / f"{dataset_name}_200k_36.f32"
            feature_data = np.frombuffer(open(f32_file, 'rb').read(), dtype=np.float32)
            feature_data = feature_data.reshape(-1, 36)

            # 加载StableCodec特征
            pt_file = self.data_dir / f"{dataset_name}_200k.pt"
            stable_codec_data = torch.load(pt_file, map_location='cpu').numpy()

            # 验证数据长度对齐
            expected_audio_len = len(feature_data) * 160  # 100fps -> 16kHz
            if len(audio_data) != expected_audio_len:
                logger.warning(f"{dataset_name}: 音频长度不匹配，截断到对齐长度")
                audio_data = audio_data[:expected_audio_len]

            if len(stable_codec_data) != len(feature_data):
                min_len = min(len(stable_codec_data), len(feature_data))
                logger.warning(f"{dataset_name}: 特征长度不匹配，截断到 {min_len}")
                stable_codec_data = stable_codec_data[:min_len]
                feature_data = feature_data[:min_len]
                audio_data = audio_data[:min_len * 160]

            self.datasets[dataset_name] = {
                'audio': audio_data,
                'features': feature_data,
                'stable_codec': stable_codec_data,
                'length': len(feature_data)
            }

            logger.info(f"  - 音频: {len(audio_data)} 样本 ({len(audio_data)/16000:.1f}秒)")
            logger.info(f"  - 特征: {feature_data.shape}")
            logger.info(f"  - StableCodec: {stable_codec_data.shape}")

        # 根据split划分数据集
        self._create_split_indices()

    def _create_split_indices(self):
        """根据split划分和dataset_ratios创建样本索引"""
        self.sample_indices = []  # [(dataset_name, start_frame, length), ...]

        # 分割比例
        split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

        # 每个数据集的帧数
        total_frames_per_dataset = 200000  # 每个数据集200k帧
        target_length = 2048  # 每个样本的长度
        overlap_ratio = 0.25  # 重叠比例
        step_size = int(target_length * (1 - overlap_ratio))

        for dataset_name, dataset_ratio in self.dataset_ratios.items():
            if dataset_name not in self.datasets:
                continue

            # 计算当前split的起止位置
            if self.split == "train":
                start_frame = 0
                end_frame = int(total_frames_per_dataset * split_ratios["train"])
            elif self.split == "val":
                start_frame = int(total_frames_per_dataset * split_ratios["train"])
                end_frame = int(total_frames_per_dataset * (split_ratios["train"] + split_ratios["val"]))
            else:  # test
                start_frame = int(total_frames_per_dataset * (split_ratios["train"] + split_ratios["val"]))
                end_frame = total_frames_per_dataset

            # 生成滑动窗口样本
            for sample_start in range(start_frame, end_frame - target_length, step_size):
                self.sample_indices.append((dataset_name, sample_start, target_length))

        # 根据数据集占比调整样本数量
        self._adjust_sample_ratios()

        logger.info(f"创建{self.split}集: {len(self.sample_indices)}个样本")
        for dataset_name in self.datasets.keys():
            count = sum(1 for idx in self.sample_indices if idx[0] == dataset_name)
            ratio = count / len(self.sample_indices) if self.sample_indices else 0
            logger.info(f"  - {dataset_name}: {count} 样本 ({ratio:.1%})")

    def _adjust_sample_ratios(self):
        """调整各数据集的样本数量以匹配目标占比"""
        if not self.sample_indices:
            return

        # 统计当前各数据集的样本数
        current_counts = {}
        for dataset_name, _, _ in self.sample_indices:
            current_counts[dataset_name] = current_counts.get(dataset_name, 0) + 1

        # 计算目标样本数
        total_samples = len(self.sample_indices)
        target_samples = {}
        for dataset_name, ratio in self.dataset_ratios.items():
            if dataset_name in current_counts:
                target_samples[dataset_name] = int(total_samples * ratio)

        # 调整样本索引
        adjusted_indices = []
        for dataset_name, target_count in target_samples.items():
            current_indices = [(i, idx) for i, idx in enumerate(self.sample_indices)
                             if idx[0] == dataset_name]

            if len(current_indices) >= target_count:
                # 随机采样到目标数量
                selected = np.random.choice(len(current_indices), target_count, replace=False)
                adjusted_indices.extend([current_indices[i][1] for i in selected])
            else:
                # 重复采样到目标数量
                selected = np.random.choice(len(current_indices), target_count, replace=True)
                adjusted_indices.extend([current_indices[i][1] for i in selected])

        # 随机打乱
        np.random.shuffle(adjusted_indices)
        self.sample_indices = adjusted_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个训练样本"""
        dataset_name, start_frame, length = self.sample_indices[idx]

        # 从对应数据集提取数据
        dataset = self.datasets[dataset_name]

        # 提取特征段 (36维)
        features = dataset['features'][start_frame:start_frame + length].copy()

        # 提取StableCodec特征 (6维)
        stable_codec = dataset['stable_codec'][start_frame:start_frame + length].copy()

        # 提取音频段
        audio_start = start_frame * 160  # 100fps -> 16kHz
        audio_end = audio_start + length * 160
        audio = dataset['audio'][audio_start:audio_end].copy()

        # 转换为张量
        audio_tensor = torch.from_numpy(audio).float()
        features_tensor = torch.from_numpy(features).float()
        stable_codec_tensor = torch.from_numpy(stable_codec).float()

        sample = {
            'audio': audio_tensor,
            'features': features_tensor,
            'stable_codec': stable_codec_tensor,
            'dataset_name': dataset_name,
            'audio_length': torch.tensor(len(audio), dtype=torch.long),
            'feature_length': torch.tensor(len(features), dtype=torch.long),
            'idx': torch.tensor(idx, dtype=torch.long)
        }

        # 应用数据增强
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_feature_stats(self) -> Optional[Dict]:
        """获取特征统计信息"""
        return None  # 暂时不支持，因为使用多数据集融合

    def get_config(self) -> Dict:
        """获取数据配置"""
        return self.config

    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        info = {
            'available_datasets': list(self.datasets.keys()),
            'dataset_ratios': self.dataset_ratios,
            'total_samples': len(self.sample_indices),
            'samples_per_dataset': {}
        }

        for dataset_name in self.datasets.keys():
            count = sum(1 for idx in self.sample_indices if idx[0] == dataset_name)
            info['samples_per_dataset'][dataset_name] = count

        return info

class AetherLiteCollator:
    """Aether-Lite数据批处理器"""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """将批次数据进行padding对齐"""
        batch_size = len(batch)

        # 获取最大长度
        max_audio_len = max(sample['audio_length'].item() for sample in batch)
        max_feat_len = max(sample['feature_length'].item() for sample in batch)

        # 创建padded张量
        audio_batch = torch.full((batch_size, max_audio_len), self.pad_value, dtype=torch.float32)
        features_batch = torch.full((batch_size, max_feat_len, 36), self.pad_value, dtype=torch.float32)
        stable_codec_batch = torch.full((batch_size, max_feat_len, 6), self.pad_value, dtype=torch.float32)

        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        feature_lengths = torch.zeros(batch_size, dtype=torch.long)
        indices = torch.zeros(batch_size, dtype=torch.long)
        dataset_names = []

        # 填充数据
        for i, sample in enumerate(batch):
            audio_len = sample['audio_length'].item()
            feat_len = sample['feature_length'].item()

            audio_batch[i, :audio_len] = sample['audio']
            features_batch[i, :feat_len] = sample['features']
            stable_codec_batch[i, :feat_len] = sample['stable_codec']
            audio_lengths[i] = audio_len
            feature_lengths[i] = feat_len
            indices[i] = sample['idx']
            dataset_names.append(sample['dataset_name'])

        # 创建mask
        audio_mask = torch.arange(max_audio_len)[None, :] < audio_lengths[:, None]
        feature_mask = torch.arange(max_feat_len)[None, :] < feature_lengths[:, None]

        return {
            'audio': audio_batch,
            'features': features_batch,
            'stable_codec': stable_codec_batch,
            'audio_lengths': audio_lengths,
            'feature_lengths': feature_lengths,
            'audio_mask': audio_mask,
            'feature_mask': feature_mask,
            'indices': indices,
            'dataset_names': dataset_names
        }

class AetherLiteTransforms:
    """Aether-Lite数据增强变换"""

    @staticmethod
    def add_gaussian_noise(snr_db: float = 30.0):
        """添加高斯噪声"""
        def transform(sample):
            audio = sample['audio']
            signal_power = torch.mean(audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.normal(0, torch.sqrt(noise_power), audio.shape)
            sample['audio'] = audio + noise
            return sample
        return transform

    @staticmethod
    def time_shift(max_shift_ms: float = 50.0, sample_rate: int = 16000):
        """时间偏移"""
        def transform(sample):
            max_shift_samples = int(max_shift_ms * sample_rate / 1000)
            shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)

            audio = sample['audio']
            if shift > 0:
                # 右移：前面填零
                audio_shifted = torch.cat([torch.zeros(shift), audio[:-shift]])
            elif shift < 0:
                # 左移：后面填零
                audio_shifted = torch.cat([audio[-shift:], torch.zeros(-shift)])
            else:
                audio_shifted = audio

            sample['audio'] = audio_shifted
            return sample
        return transform

    @staticmethod
    def amplitude_scale(scale_range: Tuple[float, float] = (0.8, 1.2)):
        """幅度缩放"""
        def transform(sample):
            scale = np.random.uniform(*scale_range)
            sample['audio'] = sample['audio'] * scale
            return sample
        return transform

    @staticmethod
    def feature_dropout(dropout_rate: float = 0.1):
        """特征dropout"""
        def transform(sample):
            features = sample['features']
            if np.random.rand() < dropout_rate:
                # 随机mask部分特征维度
                mask = torch.rand(features.shape[-1]) > dropout_rate
                features = features * mask[None, :]
                sample['features'] = features
            return sample
        return transform

def create_dataloaders(data_dir: str,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      augment_train: bool = True,
                      dataset_ratios: Optional[Dict[str, float]] = None) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""

    # 默认数据集占比
    if dataset_ratios is None:
        dataset_ratios = {
            "burst_inpaint": 0.3,  # 突发修复场景
            "harmonic": 0.3,       # 谐波丰富场景
            "low_snr": 0.2,        # 低信噪比场景
            "transient": 0.2       # 瞬态信号场景
        }

    # 训练集数据增强
    train_transforms = None
    if augment_train:
        transforms_list = [
            AetherLiteTransforms.add_gaussian_noise(snr_db=35.0),
            AetherLiteTransforms.amplitude_scale(scale_range=(0.85, 1.15)),
            AetherLiteTransforms.feature_dropout(dropout_rate=0.05)
        ]

        def compose_transforms(sample):
            for transform in transforms_list:
                if np.random.rand() < 0.3:  # 30%概率应用每个变换
                    sample = transform(sample)
            return sample

        train_transforms = compose_transforms

    # 创建数据集
    train_dataset = AetherLiteDataset(
        data_dir,
        split="train",
        transform=train_transforms,
        dataset_ratios=dataset_ratios
    )
    val_dataset = AetherLiteDataset(
        data_dir,
        split="val",
        transform=None,
        dataset_ratios=dataset_ratios
    )

    # 创建collator
    collator = AetherLiteCollator()

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False
    )

    logger.info(f"训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    logger.info(f"验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")

    # 打印数据集占比信息
    train_info = train_dataset.get_dataset_info()
    logger.info("训练集数据分布:")
    for dataset_name, count in train_info['samples_per_dataset'].items():
        ratio = count / train_info['total_samples']
        logger.info(f"  {dataset_name}: {count} 样本 ({ratio:.1%})")

    return train_loader, val_loader

def create_test_dataloader(data_dir: str,
                          batch_size: int = 8,
                          num_workers: int = 4) -> DataLoader:
    """创建测试数据加载器"""
    test_dataset = AetherLiteDataset(data_dir, split="test", transform=None)
    collator = AetherLiteCollator()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )

    logger.info(f"测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    return test_loader

# 示例使用
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    args = parser.parse_args()

    # 测试数据集
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )

    print("训练数据样本:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Features: {batch['features'].shape}")
        print(f"  Audio lengths: {batch['audio_lengths']}")
        print(f"  Feature lengths: {batch['feature_lengths']}")

        if i >= 2:  # 只看前3个batch
            break

    print("\n验证数据样本:")
    for i, batch in enumerate(val_loader):
        print(f"Batch {i}:")
        print(f"  Audio: {batch['audio'].shape}")
        print(f"  Features: {batch['features'].shape}")

        if i >= 1:
            break