#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0: 外部超大老师特征提取

基于用户技术方案的三层Teacher-Student架构：
- 层0：外部超大老师（StableCodec等）
- 用途：离线提取高质量latent和波形，为后续训练提供蒸馏目标

关键功能：
1. 使用StableCodec提取高维latent（128-256维）
2. 生成高质量重建波形wav_teacher
3. 存储语义特征用于Stage3监督
4. 为Aether-Base提供蒸馏目标
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

import torch
import torch.nn as nn
import tqdm
import soundfile as sf

# 使用本地移植的组件
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stablecodec_teacher import StableCodecTeacher
from models.semantic_extractor import create_semantic_extractor
from utils.real_data_loader import create_aether_data_loader


class ExternalTeacherProcessor:
    """外部超大老师处理器"""

    def __init__(self,
                 stablecodec_model: str = "stabilityai/stable-audio-open-1.0",
                 device: str | None = None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 初始化StableCodec teacher
        # StableCodecTeacher 使用 pretrained_model 关键字来指定预训练权重标识
        self.stablecodec = StableCodecTeacher(
            pretrained_model=stablecodec_model,
            device=self.device
        )

        # 初始化语义提取器（使用StableCodec作为SSL模型）
        self.semantic_extractor = create_semantic_extractor(
            model_name="stablecodec",  # 使用StableCodec
            proj_dim=16,
            device=self.device
        )

        print(f"✅ External Teacher initialized:")
        print(f"   StableCodec: {stablecodec_model}")
        print(f"   Semantic Extractor: StableCodec-based")
        print(f"   Device: {self.device}")

    @torch.no_grad()
    def process_audio_batch(self, audio_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        处理一个batch的音频，提取外部老师特征

        Args:
            audio_batch: [B, L] 音频波形

        Returns:
            字典包含：
            - latent_teacher: [B, T_latent, D_latent] 高维latent
            - wav_teacher: [B, L] 重建音频
            - semantic_teacher: [B, T_frames, 16] 语义特征
        """
        B, L = audio_batch.shape
        audio_batch = audio_batch.to(self.device)

        # 1. StableCodec编码-解码
        stablecodec_output = self.stablecodec.encode_decode(audio_batch)
        latent_teacher = stablecodec_output['latent']  # [B, T_latent, D_latent]
        wav_teacher = stablecodec_output['reconstructed']  # [B, L]

        # 2. 语义特征提取（使用原始音频）
        # 估算帧数：16kHz音频，100Hz帧率
        target_frames = int(L * 100 / 16000)
        semantic_teacher = self.semantic_extractor(audio_batch, target_frames=target_frames)

        return {
            'latent_teacher': latent_teacher.cpu(),
            'wav_teacher': wav_teacher.cpu(),
            'semantic_teacher': semantic_teacher.cpu()
        }

    def process_dataset(self,
                       data_loader,
                       output_dir: str,
                       max_batches: Optional[int] = None):
        """
        处理整个数据集，生成外部老师特征

        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
            max_batches: 最大处理batch数（None为全部）
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 统计信息
        stats = {
            'total_batches': 0,
            'total_samples': 0,
            'latent_dim': None,
            'semantic_dim': 16,
            'sample_rate': 16000
        }

        print(f"🚀 开始处理数据集，输出到: {output_dir}")

        for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, desc="Processing batches")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # 提取音频（假设batch包含audio）
            if isinstance(batch, dict):
                audio = batch['audio']
            else:
                audio = batch[0]  # 假设第一个元素是音频

            # 处理batch
            teacher_features = self.process_audio_batch(audio)

            # 更新统计
            stats['total_batches'] += 1
            stats['total_samples'] += audio.size(0)
            if stats['latent_dim'] is None:
                stats['latent_dim'] = teacher_features['latent_teacher'].size(-1)

            # 保存特征
            batch_output_file = output_dir / f"teacher_features_batch_{batch_idx:06d}.npz"

            # 转换为numpy保存
            features_np = {}
            for key, tensor in teacher_features.items():
                features_np[key] = tensor.numpy()

            np.savez_compressed(batch_output_file, **features_np)

            # 定期打印进度
            if batch_idx % 100 == 0:
                print(f"   Processed {batch_idx} batches, {stats['total_samples']} samples")

        # 保存统计信息
        stats_file = output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✅ 处理完成:")
        print(f"   总batch数: {stats['total_batches']}")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   Latent维度: {stats['latent_dim']}")
        print(f"   语义维度: {stats['semantic_dim']}")
        print(f"   特征文件保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 0: External Teacher Feature Extraction")
    parser.add_argument("--features", type=str, required=True, help="Input features file (.f32)")
    parser.add_argument("--pcm", type=str, required=True, help="Input PCM audio file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for teacher features")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to process (for testing)")
    parser.add_argument("--stablecodec_model", type=str,
                       default="stabilityai/stable-audio-open-1.0",
                       help="StableCodec model name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 Stage 0: External Teacher Feature Extraction")
    print("=" * 60)
    print(f"输入特征: {args.features}")
    print(f"输入音频: {args.pcm}")
    print(f"输出目录: {args.output_dir}")
    print(f"Batch大小: {args.batch_size}")
    print(f"StableCodec模型: {args.stablecodec_model}")

    # 创建数据加载器
    print("\n📂 创建数据加载器...")
    data_loader = create_aether_data_loader(
        features_file=args.features,
        pcm_file=args.pcm,
        batch_size=args.batch_size,
        shuffle=False  # Stage0不需要shuffle，保持顺序
    )

    print(f"数据加载器创建完成，总batch数: {len(data_loader)}")

    # 创建外部老师处理器
    print("\n🧠 初始化外部老师...")
    processor = ExternalTeacherProcessor(
        stablecodec_model=args.stablecodec_model,
        device=args.device
    )

    # 处理数据集
    print("\n🔄 开始特征提取...")
    processor.process_dataset(
        data_loader=data_loader,
        output_dir=args.output_dir,
        max_batches=args.max_batches
    )

    print("\n" + "=" * 60)
    print("Stage 0 completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
