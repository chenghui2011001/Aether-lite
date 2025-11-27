#!/usr/bin/env python3

"""
Aether-Lite数据预处理工具
从原始音频PCM和特征F32文件准备训练数据，支持多种配置和验证

使用方法:
python tools/prepare_aether_lite_data.py \
    --pcm-files data_cn/out_speech.pcm \
    --feature-files data_cn/out_features.f32 \
    --output-dir data_aether_lite \
    --split-ratio 0.8,0.1,0.1 \
    --target-length 2048
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """数据预处理配置"""
    sample_rate: int = 16000
    feature_dim: int = 36
    frame_rate: int = 100  # 100fps
    target_length: int = 2048  # 约20.48秒
    overlap_ratio: float = 0.25  # 25%重叠
    min_length: int = 1000  # 最小长度10秒
    normalize_features: bool = True
    add_noise: bool = False
    noise_snr_range: Tuple[float, float] = (20.0, 40.0)

class AetherLiteDataPreprocessor:
    """Aether-Lite数据预处理器"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_stats = None

    def load_pcm(self, pcm_path: str) -> np.ndarray:
        """加载PCM音频文件"""
        audio = np.frombuffer(open(pcm_path, 'rb').read(), dtype=np.int16)
        return audio.astype(np.float32) / 32767.0

    def load_features(self, feature_path: str) -> np.ndarray:
        """加载特征文件 (36维)"""
        features = np.frombuffer(open(feature_path, 'rb').read(), dtype=np.float32)
        return features.reshape(-1, self.config.feature_dim)

    def compute_feature_stats(self, feature_files: List[str]) -> Dict:
        """计算特征统计信息用于归一化"""
        logger.info("Computing feature statistics...")
        all_features = []

        for i, feat_file in enumerate(feature_files):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(feature_files)} files...")

            features = self.load_features(feat_file)
            # 随机采样避免内存溢出
            if len(features) > 1000:
                indices = np.random.choice(len(features), 1000, replace=False)
                features = features[indices]
            all_features.append(features)

        all_features = np.vstack(all_features)
        stats = {
            'mean': np.mean(all_features, axis=0).tolist(),
            'std': np.std(all_features, axis=0).tolist(),
            'min': np.min(all_features, axis=0).tolist(),
            'max': np.max(all_features, axis=0).tolist()
        }

        logger.info(f"Feature stats computed from {all_features.shape[0]} frames")
        return stats

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """特征归一化"""
        if self.feature_stats is None:
            return features

        mean = np.array(self.feature_stats['mean'])
        std = np.array(self.feature_stats['std'])
        std = np.where(std < 1e-6, 1.0, std)  # 避免除零

        return (features - mean) / std

    def add_channel_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """添加信道噪声模拟"""
        if not self.config.add_noise:
            return audio

        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise

    def segment_data(self, audio: np.ndarray, features: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """将长序列分割为训练片段"""
        # 确保时间对齐: 音频16kHz, 特征100Hz
        audio_frames_per_feature = self.config.sample_rate // self.config.frame_rate  # 160 samples per feature frame
        max_feature_frames = min(len(features), len(audio) // audio_frames_per_feature)

        # 截断到对齐长度
        features = features[:max_feature_frames]
        audio = audio[:max_feature_frames * audio_frames_per_feature]

        segments = []
        target_frames = self.config.target_length
        step_frames = int(target_frames * (1 - self.config.overlap_ratio))

        for start_frame in range(0, max_feature_frames - self.config.min_length, step_frames):
            end_frame = min(start_frame + target_frames, max_feature_frames)

            if end_frame - start_frame < self.config.min_length:
                break

            # 提取特征段
            feat_segment = features[start_frame:end_frame]

            # 提取对应音频段
            start_sample = start_frame * audio_frames_per_feature
            end_sample = end_frame * audio_frames_per_feature
            audio_segment = audio[start_sample:end_sample]

            # 可选添加噪声
            if self.config.add_noise:
                snr = np.random.uniform(*self.config.noise_snr_range)
                audio_segment = self.add_channel_noise(audio_segment, snr)

            segments.append((audio_segment, feat_segment))

        return segments

    def process_file_pair(self, pcm_file: str, feature_file: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """处理单个文件对"""
        try:
            # 加载数据
            audio = self.load_pcm(pcm_file)
            features = self.load_features(feature_file)

            # 归一化特征
            if self.config.normalize_features:
                features = self.normalize_features(features)

            # 分割为训练段
            segments = self.segment_data(audio, features)

            logger.debug(f"Processed {pcm_file}: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Error processing {pcm_file}: {e}")
            return []

def process_file_pair_worker(args):
    """多进程工作函数"""
    preprocessor, pcm_file, feature_file = args
    return preprocessor.process_file_pair(pcm_file, feature_file)

def split_dataset(segments: List[Tuple[np.ndarray, np.ndarray]],
                 split_ratio: Tuple[float, float, float]) -> Tuple[List, List, List]:
    """划分训练/验证/测试集"""
    np.random.shuffle(segments)

    n_total = len(segments)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_set = segments[:n_train]
    val_set = segments[n_train:n_train + n_val]
    test_set = segments[n_train + n_val:]

    return train_set, val_set, test_set

def save_dataset_split(segments: List[Tuple[np.ndarray, np.ndarray]],
                      output_dir: Path, split_name: str):
    """保存数据集分割"""
    split_dir = output_dir / split_name
    split_dir.mkdir(exist_ok=True)

    audio_file = split_dir / f"{split_name}_audio.npy"
    features_file = split_dir / f"{split_name}_features.npy"
    metadata_file = split_dir / f"{split_name}_metadata.json"

    # 分离音频和特征
    audios = [seg[0] for seg in segments]
    features = [seg[1] for seg in segments]

    # 保存为numpy数组 (使用padding对齐)
    max_audio_len = max(len(a) for a in audios) if audios else 0
    max_feat_len = max(len(f) for f in features) if features else 0

    if audios:
        audio_padded = np.zeros((len(audios), max_audio_len), dtype=np.float32)
        feat_padded = np.zeros((len(features), max_feat_len, 36), dtype=np.float32)
        lengths = []

        for i, (audio, feat) in enumerate(segments):
            audio_len = len(audio)
            feat_len = len(feat)
            audio_padded[i, :audio_len] = audio
            feat_padded[i, :feat_len] = feat
            lengths.append({'audio_len': audio_len, 'feat_len': feat_len})

        np.save(audio_file, audio_padded)
        np.save(features_file, feat_padded)

        metadata = {
            'num_samples': len(segments),
            'max_audio_length': max_audio_len,
            'max_feature_length': max_feat_len,
            'lengths': lengths
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {split_name}: {len(segments)} samples")

def main():
    parser = argparse.ArgumentParser(description="Aether-Lite数据预处理")
    parser.add_argument("--pcm-files", type=str, required=True,
                       help="PCM音频文件路径 (支持glob模式)")
    parser.add_argument("--feature-files", type=str, required=True,
                       help="特征文件路径 (支持glob模式)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1",
                       help="训练/验证/测试集比例")
    parser.add_argument("--target-length", type=int, default=2048,
                       help="目标段长度 (帧数)")
    parser.add_argument("--overlap-ratio", type=float, default=0.25,
                       help="段重叠比例")
    parser.add_argument("--add-noise", action="store_true",
                       help="添加信道噪声")
    parser.add_argument("--noise-snr-range", type=str, default="20.0,40.0",
                       help="噪声SNR范围 (dB)")
    parser.add_argument("--workers", type=int, default=4,
                       help="并行处理进程数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 解析参数
    split_ratio = [float(x) for x in args.split_ratio.split(',')]
    assert len(split_ratio) == 3 and abs(sum(split_ratio) - 1.0) < 1e-6

    noise_snr_range = [float(x) for x in args.noise_snr_range.split(',')]

    # 创建配置
    config = DataConfig(
        target_length=args.target_length,
        overlap_ratio=args.overlap_ratio,
        add_noise=args.add_noise,
        noise_snr_range=tuple(noise_snr_range)
    )

    # 查找文件
    import glob
    pcm_files = sorted(glob.glob(args.pcm_files))
    feature_files = sorted(glob.glob(args.feature_files))

    if len(pcm_files) != len(feature_files):
        logger.error(f"文件数量不匹配: {len(pcm_files)} PCM vs {len(feature_files)} features")
        return

    logger.info(f"找到 {len(pcm_files)} 对文件")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 创建预处理器
    preprocessor = AetherLiteDataPreprocessor(config)

    # 计算特征统计
    if config.normalize_features:
        preprocessor.feature_stats = preprocessor.compute_feature_stats(feature_files)

        # 保存统计信息
        with open(output_dir / "feature_stats.json", 'w') as f:
            json.dump(preprocessor.feature_stats, f, indent=2)

    # 并行处理文件
    logger.info("开始数据预处理...")

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            worker_args = [(preprocessor, pcm_f, feat_f)
                          for pcm_f, feat_f in zip(pcm_files, feature_files)]

            results = pool.map(process_file_pair_worker, worker_args)
    else:
        results = []
        for pcm_f, feat_f in zip(pcm_files, feature_files):
            results.append(preprocessor.process_file_pair(pcm_f, feat_f))

    # 合并所有段
    all_segments = []
    for segments in results:
        all_segments.extend(segments)

    logger.info(f"总共处理得到 {len(all_segments)} 个训练段")

    if not all_segments:
        logger.error("没有处理到任何有效数据")
        return

    # 划分数据集
    train_set, val_set, test_set = split_dataset(all_segments, tuple(split_ratio))

    # 保存数据集
    save_dataset_split(train_set, output_dir, "train")
    save_dataset_split(val_set, output_dir, "val")
    save_dataset_split(test_set, output_dir, "test")

    # 保存配置
    config_dict = {
        'sample_rate': config.sample_rate,
        'feature_dim': config.feature_dim,
        'frame_rate': config.frame_rate,
        'target_length': config.target_length,
        'overlap_ratio': config.overlap_ratio,
        'min_length': config.min_length,
        'normalize_features': config.normalize_features,
        'add_noise': config.add_noise,
        'noise_snr_range': config.noise_snr_range,
        'split_ratio': split_ratio
    }

    with open(output_dir / "data_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 统计信息
    logger.info("数据预处理完成!")
    logger.info(f"训练集: {len(train_set)} 样本")
    logger.info(f"验证集: {len(val_set)} 样本")
    logger.info(f"测试集: {len(test_set)} 样本")
    logger.info(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()