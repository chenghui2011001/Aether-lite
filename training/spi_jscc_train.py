#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPI-JSCC训练脚本：语义映射+位置编码+图像codec的完整训练

特点：
- 集成SPI架构的端到端训练
- 支持渐进式训练策略
- 专用的SPI损失函数
- 音频验证和检查点保存
"""

from __future__ import annotations

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchaudio
import numpy as np
from tqdm import tqdm

from utils.multi_stft_discriminator import create_adversarial_wave_loss
from models.spi_lite_jscc import SPI_LiteSpeechJSCC
from utils.channel_sim import ChannelSimulator
from utils.real_data_loader import create_combined_data_loader, create_aether_data_loader
from training.fargan_losses import multi_resolution_stft_loss
from training.enhanced_losses import create_enhanced_audio_loss
from utils.audio_visualizer import create_batch_comparison_plots, save_comparison_audio_samples


@dataclass
class SPITrainConfig:
    """SPI训练配置"""

    data_root: str = "/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/data_expert_augmented_small200k"
    stage: int = 2  # 2: 连续JSCC; 3: Hash + JSCC; 4: Hash + bit-level noise
    enable_hash: bool = False  # Step1策略：先关闭Hash，测试SPI+JSCC baseline
    # FARGAN 预训练权重
    fargan_ckpt: Optional[str] = "/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/fargan_pt/fargan_sq1Ab_adv_50.pth"
    fargan_unfreeze_steps: int = 5000  # FARGAN解冻步数
    batch_size: int = 8
    sequence_length: int = 200
    num_epochs: int = 5
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查点保存
    output_dir: str = "./checkpoints_spi_jscc"
    save_every_steps: int = 500
    resume_from: Optional[str] = None

    # 音频验证保存
    save_audio: bool = True
    audio_save_dir: str = "./audio_samples_spi"
    save_audio_every_steps: int = 100
    max_audio_samples: int = 4

    # 可视化保存
    save_visualization: bool = True
    visualization_save_dir: str = "./visualizations_spi"
    save_visualization_every_steps: int = 200
    max_visualization_samples: int = 3

    # SPI专用参数
    img_size: int = 64
    semantic_dim: int = 16
    d_z: int = 32  # 扩大latent空间
    d_s: int = 24  # 对应调整symbol空间

    # 损失权重 - 干净baseline配置
    # Step 1 baseline: 只保留 STFT + 特征重建，关闭增强和对抗
    lambda_wave: float = 1.0      # STFT损失权重 (主要损失)
    lambda_stft: float = 1.0      # STFT损失权重
    lambda_adv: float = 0.0       # 对抗损失权重 (关闭以建立baseline)
    lambda_spi: float = 0.5       # SPI专用损失权重

    # 渐进式训练
    use_progressive: bool = True  # 启用渐进式训练
    stage1_steps: int = 200       # Stage1: 无JSCC步数 (减小)
    stage2_steps: int = 400       # Stage2: 无信道噪声步数 (减小)
    stage3_steps: int = 600       # Stage3: Hash bottleneck步数 (减小)

    # 损失函数选择 - 干净baseline配置
    use_stft_loss: bool = True    # 使用STFT损失
    use_enhanced_losses: bool = False  # 关闭增强损失以建立baseline
    f0_loss_weight: float = 1.0      # F0损失权重(用于后续实验)
    perceptual_loss_weight: float = 0.1  # 感知损失权重(用于后续实验)

    # 通道配置
    snr_min_db: float = -5.0
    snr_max_db: float = 15.0

    # 多GPU支持
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1


def save_checkpoint(
    model: SPI_LiteSpeechJSCC,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    cfg: SPITrainConfig,
    epoch: int,
    global_step: int,
    output_dir: str,
) -> None:
    """保存检查点"""
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"spi_checkpoint_step_{global_step:06d}.pth")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": {
            "stage": cfg.stage,
            "img_size": cfg.img_size,
            "semantic_dim": cfg.semantic_dim,
            "d_z": cfg.d_z,
            "d_s": cfg.d_s,
            "use_progressive": cfg.use_progressive,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"SPI Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: SPI_LiteSpeechJSCC,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    """加载检查点"""
    print(f"Loading SPI checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 兼容性加载：处理模型结构变化
    model_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()

    # 过滤出匹配的参数
    filtered_state = {}
    for key, value in model_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            print(f"Skipping parameter {key}: shape mismatch or not found")

    # 加载匹配的参数
    model.load_state_dict(filtered_state, strict=False)

    # 报告加载状态
    loaded_params = len(filtered_state)
    total_params = len(model_state)
    print(f"Loaded {loaded_params}/{total_params} parameters from checkpoint")
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded successfully")
    except (ValueError, KeyError) as e:
        print(f"Cannot load optimizer state (will restart): {e}")
        print("This is normal when model structure changes")

    try:
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
        print("Discriminator optimizer state loaded successfully")
    except (ValueError, KeyError) as e:
        print(f"Cannot load discriminator optimizer state (will restart): {e}")
        print("This is normal when model structure changes")

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    print(f"Resumed from epoch {epoch}, step {global_step}")
    return epoch, global_step


def save_audio_samples(
    audio_real: torch.Tensor,
    audio_gen: torch.Tensor,
    cfg: SPITrainConfig,
    epoch: int,
    global_step: int,
    sample_rate: int = 16000,
) -> None:
    """保存音频样本"""
    if not cfg.save_audio:
        return

    os.makedirs(cfg.audio_save_dir, exist_ok=True)

    batch_size = min(audio_real.size(0), cfg.max_audio_samples)

    for i in range(batch_size):
        real_sample = audio_real[i].detach().cpu()
        gen_sample = audio_gen[i].detach().cpu()

        min_len = min(real_sample.size(0), gen_sample.size(0))
        real_sample = real_sample[:min_len]
        gen_sample = gen_sample[:min_len]

        # 归一化
        real_sample = real_sample / (real_sample.abs().max() + 1e-8)
        gen_sample = gen_sample / (gen_sample.abs().max() + 1e-8)

        # 保存文件
        real_path = os.path.join(
            cfg.audio_save_dir,
            f"spi_step_{global_step:06d}_sample_{i:02d}_real.wav"
        )
        gen_path = os.path.join(
            cfg.audio_save_dir,
            f"spi_step_{global_step:06d}_sample_{i:02d}_gen.wav"
        )

        torchaudio.save(real_path, real_sample.unsqueeze(0), sample_rate)
        torchaudio.save(gen_path, gen_sample.unsqueeze(0), sample_rate)

    print(f"SPI Audio samples saved: {batch_size} pairs at step {global_step}")


def determine_training_stage(global_step: int, cfg: SPITrainConfig) -> str:
    """确定当前训练阶段"""
    if not cfg.use_progressive:
        return f"stage{cfg.stage}"

    if global_step < cfg.stage1_steps:
        return "no_jscc"
    elif global_step < cfg.stage1_steps + cfg.stage2_steps:
        return "no_channel"
    elif global_step < cfg.stage1_steps + cfg.stage2_steps + cfg.stage3_steps:
        return "stage3_hash"
    else:
        return "stage4_hash_noise"


def forward_spi_progressive(
    model,  # 可能是 SPI_LiteSpeechJSCC 或 DistributedDataParallel
    feats: torch.Tensor,
    audio: torch.Tensor,
    channel_sim: ChannelSimulator,
    stage: str,
    cfg: SPITrainConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    SPI渐进式前向传播

    Args:
        model: SPI模型
        feats: [B, T, 20] 输入特征
        audio: [B, L] 音频
        channel_sim: 信道模拟器
        stage: 训练阶段
        cfg: 配置
        device: 设备

    Returns:
        包含所有输出的字典
    """
    B, T, _ = feats.shape

    # 生成CSI
    csi_dict, amp_t, snr_db_t = channel_sim.sample_csi(
        B, T, channel="fading", snr_min_db=cfg.snr_min_db, snr_max_db=cfg.snr_max_db
    )
    csi_vec = torch.stack([
        csi_dict["snr_proxy"],
        csi_dict["time_selectivity"],
        csi_dict["freq_selectivity"],
        csi_dict["los_ratio"],
    ], dim=-1).to(device=device, dtype=feats.dtype)

    # 获取实际模型 (处理DDP包装)
    actual_model = model.module if hasattr(model, 'module') else model

    if stage == "no_jscc":
        # Stage 1: 直接编码-解码，不经过JSCC
        encode_out = actual_model.spi_encode(feats, csi_vec)
        decode_out = actual_model.spi_decode(encode_out['z_encoded_multi'], csi_vec, encode_out['semantic_vec'], encode_out.get('temporal_info'))

        output = {
            **encode_out,
            **decode_out,
            'stage': stage,
            'feats_hat': decode_out['feats_recovered'],
        }

    elif stage == "no_channel":
        # Stage 2: 多token JSCC但无信道噪声 - 修复：真正的多token处理
        encode_out = actual_model.spi_encode(feats, csi_vec)
        z_multi = encode_out['z_encoded_multi']  # [B, N_patches, d_z]
        z = encode_out['z_encoded']              # [B, d_z] 用于兼容
        semantic_vec = encode_out['semantic_vec']

        # 多token JSCC 编码解码（不加噪声）
        s_multi = actual_model.jscc_enc(z_multi, csi_vec)        # [B, N_patches, d_s]
        z_hat_multi = actual_model.jscc_dec(s_multi, csi_vec)    # [B, N_patches, d_z]

        # 单token作为summary（用于兼容/日志）
        z_hat = z_hat_multi.mean(dim=1) if z_hat_multi.size(1) > 1 else z_hat_multi.squeeze(1)

        decode_out = actual_model.spi_decode(
            z_hat_multi,                         # 👈 使用完整多token
            csi_vec,
            semantic_vec,
            encode_out.get('temporal_info'),
            z_decoded_single=z_hat               # 单token作为备用
        )
        feat_hat = decode_out['feats_recovered']

        output = {
            **encode_out,
            **decode_out,
            'z': z,
            'z_hat': z_hat,
            'z_multi': z_multi,
            'z_hat_multi': z_hat_multi,
            's_multi': s_multi,
            'stage': stage,
            'feats_hat': feat_hat,
        }

    elif stage == "stage3_hash":
        # Stage 3: 多token JSCC + 轻量channel噪声，准备接入Hash（Step1策略：先稳定JSCC）
        encode_out = actual_model.spi_encode(feats, csi_vec)
        z_multi = encode_out['z_encoded_multi']  # [B, N_patches, d_z]
        z = encode_out['z_encoded']              # [B, d_z] 用于兼容
        semantic_vec = encode_out['semantic_vec']

        # 多token JSCC 编码
        s_multi = actual_model.jscc_enc(z_multi, csi_vec)   # [B, N_patches, d_s]

        # 轻量信道噪声（比原来温和）
        csi_dict2, amp_t2, snr_db_t2 = channel_sim.sample_csi(
            B, z_multi.size(1), channel="fading", snr_min_db=cfg.snr_min_db, snr_max_db=cfg.snr_max_db
        )
        amp_t2 = amp_t2.to(device=s_multi.device, dtype=s_multi.dtype)
        snr_db_t2 = snr_db_t2.to(device=s_multi.device, dtype=s_multi.dtype)
        s_multi_noisy = channel_sim.apply(s_multi, amp_t2, snr_db_t2)  # [B, N_patches, d_s]

        # 多token JSCC 解码
        z_hat_multi = actual_model.jscc_dec(s_multi_noisy, csi_vec)       # [B, N_patches, d_z]
        z_hat = z_hat_multi.mean(dim=1) if z_hat_multi.size(1) > 1 else z_hat_multi.squeeze(1)  # [B, d_z]

        # Hash bottleneck (no bit noise) - 在JSCC处理后的多token上操作
        hash_output = None  # 提前定义，避免未启用hash时NameError

        if hasattr(model, 'hash') and actual_model.hash is not None:
            # 使用JSCC处理后的多token latents，而不是原始编码
            if z_hat_multi is not None:
                # 获取patch mask
                temporal_info = encode_out.get('temporal_info')
                patch_mask = temporal_info.get('patch_mask') if temporal_info else None
                hash_output = actual_model.hash(z_hat_multi, channel_params=None, mask=patch_mask)  # [B, N_patches, d_z]
                z_hash_multi = hash_output['reconstructed']  # [B, N_patches, d_z]
                hash_bits_clean = hash_output['hash_bits_clean']  # [B, N_patches, hash_bits]
                hash_logits = hash_output['hash_logits']  # [B, N_patches, hash_bits]
                # 为了兼容单token接口，使用第一个patch作为代表
                z_hash = z_hash_multi[:, 0, :]  # [B, d_z]
                hash_mask = hash_output.get('mask')
            else:
                # 回退到单token处理
                hash_output = actual_model.hash(z_hat.unsqueeze(1), channel_params=None)  # [B, 1, d_z]
                z_hash = hash_output['reconstructed'].squeeze(1)  # [B, d_z]
                hash_bits_clean = hash_output['hash_bits_clean'].squeeze(1)  # [B, hash_bits]
                hash_logits = hash_output['hash_logits'].squeeze(1)  # [B, hash_bits]
                hash_mask = hash_output.get('mask')
        else:
            z_hash = z_hat
            z_hash_multi = encode_out.get('z_encoded_multi', z_hat.unsqueeze(1))
            hash_bits_clean = None
            hash_logits = None
            hash_mask = None

        # SPI 解码 - 修复：使用完整多token而不是只用第一个
        decode_out = actual_model.spi_decode(
            z_hash_multi,                    # 👈 使用完整多token作为主干
            csi_vec,
            semantic_vec,
            encode_out.get('temporal_info'),
            z_decoded_single=z_hash          # 单token作为备用，不是主干
        )
        feat_hat = decode_out['feats_recovered']

        output = {
            **encode_out,
            **decode_out,
            'z': z,
            'z_hat': z_hat,
            'z_multi': z_multi,
            'z_hat_multi': z_hat_multi,
            'z_hash': z_hash,
            'z_hash_multi': z_hash_multi,
            's_multi': s_multi,
            's_multi_noisy': s_multi_noisy,
            'hash_bits_clean': hash_bits_clean,
            'hash_logits': hash_logits,
            'hash_mask': hash_mask,
            'stage': stage,
            'feats_hat': feat_hat,
            'actual_snr': 10 * torch.log10(
                torch.mean(s_multi.pow(2)) / (torch.mean((s_multi_noisy - s_multi).pow(2)) + 1e-8)
            ),
        }

    elif stage == "stage4_hash_noise":
        # Stage 4: 多token JSCC + 信道噪声 + Hash + bit噪声 (完全串联)
        encode_out = actual_model.spi_encode(feats, csi_vec)
        z_multi = encode_out['z_encoded_multi']  # [B, N_patches, d_z]
        semantic_vec = encode_out['semantic_vec']
        temporal_info = encode_out.get('temporal_info')

        # 1) 多token JSCC 编码
        s_multi = actual_model.jscc_enc(z_multi, csi_vec)    # [B, N_patches, d_s]

        # 2) 多token 信道噪声
        csi_dict2, amp_t2, snr_db_t2 = channel_sim.sample_csi(
            B, z_multi.size(1),
            channel="fading",
            snr_min_db=cfg.snr_min_db,
            snr_max_db=cfg.snr_max_db
        )
        amp_t2 = amp_t2.to(device=s_multi.device, dtype=s_multi.dtype)
        snr_db_t2 = snr_db_t2.to(device=s_multi.device, dtype=s_multi.dtype)
        s_multi_noisy = channel_sim.apply(s_multi, amp_t2, snr_db_t2)

        # 3) 多token JSCC 解码
        z_hat_multi = actual_model.jscc_dec(s_multi_noisy, csi_vec)   # [B, N_patches, d_z]

        # 4) Hash bottleneck + bit 噪声（在 JSCC 输出上）
        hash_output = None  # 提前定义，避免未启用hash时NameError

        if hasattr(model, 'hash') and actual_model.hash is not None:
            base_ber = 0.01
            max_ber = 0.15
            warm_frac = 0.3  # 简单先写死
            scheduled_ber = base_ber + warm_frac * (max_ber - base_ber)
            channel_params = {'ber': scheduled_ber}

            patch_mask = temporal_info.get('patch_mask') if temporal_info else None
            hash_output = actual_model.hash(z_hat_multi, channel_params=channel_params, mask=patch_mask)
            z_hash_multi = hash_output['reconstructed']
            hash_bits_clean = hash_output['hash_bits_clean']
            hash_bits_noisy = hash_output['hash_bits_noisy']
            hash_logits = hash_output['hash_logits']
            hash_mask = hash_output.get('mask')
        else:
            z_hash_multi = z_hat_multi
            hash_bits_clean = hash_bits_noisy = hash_logits = None
            hash_mask = None

        # 5) SPI 解码（多token主干）
        decode_out = actual_model.spi_decode(
            z_hash_multi,
            csi_vec,
            semantic_vec,
            temporal_info
        )
        feat_hat = decode_out['feats_recovered']

        output = {
            **encode_out,
            **decode_out,
            'z_multi': z_multi,
            'z_hat_multi': z_hat_multi,
            'z_hash_multi': z_hash_multi,
            's_multi': s_multi,
            's_multi_noisy': s_multi_noisy,
            'hash_bits_clean': hash_bits_clean,
            'hash_bits_noisy': hash_bits_noisy,
            'hash_logits': hash_logits,
            'hash_mask': hash_mask,
            'stage': stage,
            'feats_hat': feat_hat,
            'actual_snr': 10 * torch.log10(
                torch.mean(s_multi.pow(2)) / (torch.mean((s_multi_noisy - s_multi).pow(2)) + 1e-8)
            ),
        }

    # 通过vocoder生成音频
    feat_hat = output['feats_hat']
    target_len = audio.size(-1)

    try:
        period, audio_hat = actual_model.vocoder(feat_hat, target_len=target_len)
        audio_hat = audio_hat.squeeze(1) if audio_hat.dim() > 2 else audio_hat

        # 长度对齐
        if audio_hat.size(-1) != audio.size(-1):
            min_len = min(audio_hat.size(-1), audio.size(-1))
            audio_hat = audio_hat[..., :min_len]
            audio = audio[..., :min_len]

        output.update({
            'audio_hat': audio_hat,
            'audio_real': audio,
            'period': period,
        })

    except Exception as e:
        print(f"Vocoder error: {e}")
        # 创建dummy音频
        audio_hat = torch.zeros_like(audio)
        output.update({
            'audio_hat': audio_hat,
            'audio_real': audio,
            'period': None,
        })

    return output


class ProgressiveLossScheduler:
    """
    渐进式损失权重调度器 - Anti-Buzz训练策略

    实现分阶段的损失权重调整策略：
    - Stage 1 (0-1000): 优先特征级监督，关闭GAN
    - Stage 2 (1000-3000): 加入音频级监督，弱GAN
    - Stage 3 (3000-5000): 逐步恢复GAN权重
    - Stage 4 (5000+): 解冻FARGAN，端到端优化
    """

    def __init__(
        self,
        stage1_steps: int = 1000,
        stage2_steps: int = 3000,
        stage3_steps: int = 5000,
        max_adv_weight: float = 0.3,
        max_spi_weight: float = 1.0
    ):
        self.stage1_steps = stage1_steps
        self.stage2_steps = stage2_steps
        self.stage3_steps = stage3_steps
        self.max_adv_weight = max_adv_weight
        self.max_spi_weight = max_spi_weight

    def get_weights(self, global_step: int) -> Dict[str, float]:
        """
        根据训练步数返回当前的损失权重

        Args:
            global_step: 当前训练步数

        Returns:
            权重字典
        """
        weights = {
            'lambda_adv': 0.0,
            'lambda_spi': self.max_spi_weight,
            'lambda_wave': 1.0,
            'fargan_freeze': True,
            'stage_name': 'unknown'
        }

        if global_step < self.stage1_steps:
            # Stage 1: 优先特征级监督
            weights.update({
                'lambda_adv': 0.0,
                'lambda_spi': self.max_spi_weight,
                'fargan_freeze': True,
                'stage_name': 'feature_focus'
            })

        elif global_step < self.stage2_steps:
            # Stage 2: 加入音频级监督，弱GAN
            weights.update({
                'lambda_adv': 0.1 * self.max_adv_weight,
                'lambda_spi': self.max_spi_weight,
                'fargan_freeze': True,
                'stage_name': 'audio_weak_gan'
            })

        elif global_step < self.stage3_steps:
            # Stage 3: 逐步恢复GAN权重
            progress = (global_step - self.stage2_steps) / (self.stage3_steps - self.stage2_steps)
            weights.update({
                'lambda_adv': progress * self.max_adv_weight,
                'lambda_spi': self.max_spi_weight,
                'fargan_freeze': True,
                'stage_name': 'progressive_gan'
            })

        else:
            # Stage 4: 解冻FARGAN，端到端优化
            weights.update({
                'lambda_adv': self.max_adv_weight,
                'lambda_spi': self.max_spi_weight * 0.8,  # 略微降低SPI权重
                'fargan_freeze': False,
                'stage_name': 'end_to_end'
            })

        return weights


def compute_spi_training_loss(
    output: Dict,
    cfg: SPITrainConfig,
    adv_wave_loss,
    enhanced_loss,
    device: torch.device,
    model = None,  # 可能是 SPI_LiteSpeechJSCC 或 DistributedDataParallel
    global_step: int = 0,
    loss_scheduler: Optional[ProgressiveLossScheduler] = None
) -> Tuple[torch.Tensor, Dict]:
    """计算SPI训练损失（集成Anti-Buzz和渐进式调度）"""

    audio_hat = output['audio_hat']
    audio_real = output['audio_real']
    feats_hat = output['feats_hat']
    feats = output.get('feats', None)

    # 获取实际模型 (处理DDP包装)
    actual_model = model.module if hasattr(model, 'module') else model

    losses = {}

    # 获取渐进式权重
    if loss_scheduler is not None:
        schedule_weights = loss_scheduler.get_weights(global_step)
        current_lambda_adv = schedule_weights['lambda_adv']
        current_lambda_spi = schedule_weights['lambda_spi']
        stage_name = schedule_weights['stage_name']
    else:
        current_lambda_adv = cfg.lambda_adv
        current_lambda_spi = cfg.lambda_spi
        stage_name = 'default'

    # 1. 基础音频损失
    if cfg.use_stft_loss:
        # 多分辨率STFT损失
        loss_stft = multi_resolution_stft_loss(
            audio_hat, audio_real,
            device=device,
            fft_sizes=[1024, 512, 256, 128],
            hop_sizes=[256, 128, 64, 32],
            win_lengths=[1024, 512, 256, 128]
        )
        losses['recon'] = loss_stft
        recon_type = "stft"
    else:
        # 传统L1损失
        losses['recon'] = F.l1_loss(audio_hat, audio_real)
        recon_type = "l1"

    # 2. 增强音频损失 (F0 + 感知损失) - 仅在Stage2+启用
    if enhanced_loss is not None and cfg.use_enhanced_losses and global_step >= 1000:
        try:
            enhanced_losses = enhanced_loss(audio_hat, audio_real)

            # 将各组件损失添加到总损失中
            for key, value in enhanced_losses.items():
                if key != 'enhanced_total':  # 避免重复计算总损失
                    losses[f'enhanced_{key}'] = value

            # 总的增强损失
            losses['enhanced'] = enhanced_losses['enhanced_total']

        except Exception as e:
            print(f"Enhanced loss computation failed: {e}")
            losses['enhanced'] = torch.zeros(1, device=device)
    else:
        losses['enhanced'] = torch.zeros(1, device=device)

    # 3. 对抗损失 (根据调度器控制)
    if adv_wave_loss is not None and current_lambda_adv > 0:
        try:
            gen_out = adv_wave_loss.generator_step(audio_real, audio_hat)
            losses['adv'] = gen_out["total_adversarial_loss"]
        except:
            losses['adv'] = torch.zeros(1, device=device)
    else:
        losses['adv'] = torch.zeros(1, device=device)

    # 4. SPI专用损失（集成Anti-Buzz功能）
    if feats is not None and model is not None:
        # 使用模型的集成Anti-Buzz损失计算
        spi_loss_total, spi_losses = actual_model.compute_spi_loss(
            output, feats, audio_hat, audio_real
        )
        losses['spi_total'] = spi_loss_total

        # 添加详细的SPI损失组件
        for key, value in spi_losses.items():
            if key != 'spi_total':
                losses[f'spi_{key}'] = value

    else:
        # 备用：简单特征重建损失
        if feats is not None:
            losses['feat_recon'] = F.mse_loss(feats_hat, feats)
            losses['spi_total'] = losses['feat_recon']
        else:
            losses['spi_total'] = torch.zeros(1, device=device)

    # 5. Hash bottleneck正则化损失 (如果有hash相关输出)
    hash_logits = output.get('hash_logits')
    hash_bits_clean = output.get('hash_bits_clean')
    hash_mask = output.get('hash_mask')

    if hash_logits is not None and hash_bits_clean is not None and hasattr(model, 'hash') and actual_model.hash is not None:
        # 使用HashBottleneck的新API计算正则化损失，支持mask
        hash_reg_losses = actual_model.hash.compute_hash_regularization(
            hash_logits, hash_bits_clean, mask=hash_mask
        )

        # 提取各项损失
        losses['bit_balance'] = hash_reg_losses['bit_balance']
        losses['quantization'] = hash_reg_losses['quantization']
        losses['entropy'] = hash_reg_losses['entropy']
        losses['rate_kl'] = hash_reg_losses['rate_kl']
        losses['bit_decorrelation'] = hash_reg_losses.get('bit_decorrelation', torch.tensor(0.0, device=device))

        # 根据训练阶段调整hash权重
        if stage_name == 'feature_focus':
            hash_weight = 0.02  # Stage1：最小hash权重，专注特征重建
        elif stage_name in ['audio_weak_gan', 'progressive_gan']:
            hash_weight = 0.05  # Stage2-3：适中hash权重
        else:
            hash_weight = 0.1   # Stage4：恢复正常hash权重

        losses['hash_total'] = (
            hash_weight * losses['bit_balance'] +
            hash_weight * losses['quantization'] +
            hash_weight * losses['entropy'] +
            0.05 * losses['rate_kl'] +
            0.02 * losses['bit_decorrelation']
        )

    else:
        losses['bit_balance'] = torch.zeros(1, device=device)
        losses['quantization'] = torch.zeros(1, device=device)
        losses['entropy'] = torch.zeros(1, device=device)
        losses['rate_kl'] = torch.zeros(1, device=device)
        losses['bit_decorrelation'] = torch.zeros(1, device=device)
        losses['hash_total'] = torch.zeros(1, device=device)

    # 6. 自动缩放对抗损失（仅在启用时）
    eps = 1e-6
    if current_lambda_adv > 0 and losses['adv'].item() > eps:
        adv_scale = (losses['recon'].detach() + eps) / (losses['adv'].detach() + eps)
        adv_scale = torch.clamp(adv_scale, 0.001, 0.1)
    else:
        adv_scale = torch.tensor(0.001, device=device)

    # 7. 总损失（使用渐进式权重）
    total_loss = (
        cfg.lambda_wave * losses['recon'] +
        current_lambda_adv * adv_scale * losses['adv'] +
        current_lambda_spi * losses.get('spi_total', torch.zeros(1, device=device)) +
        losses['enhanced'] +  # 增强损失
        losses.get('hash_total', torch.zeros(1, device=device))  # Hash正则化
    )

    # 添加元数据
    losses.update({
        'total': total_loss,
        'adv_scale': adv_scale,
        'recon_type': recon_type,
        'stage_name': stage_name,
        'current_lambda_adv': current_lambda_adv,
        'current_lambda_spi': current_lambda_spi,
    })

    return total_loss, losses


def build_spi_model(cfg: SPITrainConfig) -> SPI_LiteSpeechJSCC:
    """构建SPI模型"""
    device = torch.device(cfg.device)

    model = SPI_LiteSpeechJSCC(
        feat_dim=20,  # 只使用声学特征
        d_csi=4,
        d_z=cfg.d_z,
        d_s=cfg.d_s,
        n_bits=cfg.d_s,
        hidden=80,
        img_size=cfg.img_size,
        semantic_dim=cfg.semantic_dim,
        device=device,
    )

    # Step1策略：根据配置决定是否启用Hash
    if not cfg.enable_hash:
        print("Hash bottleneck disabled for clean baseline testing")
        model.hash = None  # 禁用Hash模块

    return model


def init_fargan_from_checkpoint(model, cfg: SPITrainConfig, device: torch.device) -> None:
    """加载并冻结FARGAN声码器权重（只修改本脚本内的使用方式）"""
    # 获取实际模型 (处理DDP包装)
    actual_model = model.module if hasattr(model, 'module') else model

    if not hasattr(actual_model, "vocoder"):
        print("Model has no vocoder; skip FARGAN init.")
        return

    ckpt_path = cfg.fargan_ckpt
    if not ckpt_path:
        print("No FARGAN checkpoint path provided; skip FARGAN init.")
        return

    if not os.path.isfile(ckpt_path):
        print(f"FARGAN checkpoint not found: {ckpt_path}")
        return

    print(f"Loading FARGAN weights from: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"Failed to load FARGAN checkpoint: {e}")
        return

    # 尝试从常见字段提取解码器权重
    if isinstance(ckpt, dict):
        if "decoder_state_dict" in ckpt:
            dec_sd = ckpt["decoder_state_dict"]
        elif "model_state_dict" in ckpt:
            dec_sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            dec_sd = ckpt["state_dict"]
        else:
            dec_sd = ckpt
    else:
        dec_sd = ckpt

    voc_sd = actual_model.vocoder.state_dict()
    to_load = {}

    # 映射checkpoint参数名到模型参数名
    for ckpt_name, ckpt_param in dec_sd.items():
        if not isinstance(ckpt_param, torch.Tensor):
            continue

        # 尝试添加fargan_core前缀进行匹配
        model_name = f"fargan_core.{ckpt_name}"

        if model_name in voc_sd and ckpt_param.shape == voc_sd[model_name].shape:
            to_load[model_name] = ckpt_param
            print(f"  Mapping: {ckpt_name} -> {model_name} {ckpt_param.shape}")

    if not to_load:
        print("No matching FARGAN parameters found in checkpoint; vocoder left unchanged.")
    else:
        voc_sd.update(to_load)
        actual_model.vocoder.load_state_dict(voc_sd, strict=False)
        print(f"Successfully loaded {len(to_load)} FARGAN parameters into vocoder!")

    # 冻结FARGAN声码器参数
    frozen_count = 0
    for _, param in actual_model.vocoder.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
    print(f"FARGAN vocoder parameters frozen (count={frozen_count}).")


def unfreeze_fargan_vocoder(model, global_step: int, loss_scheduler: ProgressiveLossScheduler) -> bool:
    """根据渐进式调度器解冻FARGAN声码器 (Stage 4: 5000+ steps)"""
    # 获取实际模型 (处理DDP包装)
    actual_model = model.module if hasattr(model, 'module') else model

    if not hasattr(actual_model, "vocoder") or loss_scheduler is None:
        return False

    # 检查是否进入Stage 4 (fargan_freeze = False)
    weights = loss_scheduler.get_weights(global_step)
    should_unfreeze = not weights['fargan_freeze']

    # 仅在第一次进入Stage 4时解冻
    if should_unfreeze and global_step == loss_scheduler.stage3_steps:
        unfrozen_count = 0
        for _, param in actual_model.vocoder.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1

        if unfrozen_count > 0:
            print(f"FARGAN vocoder parameters unfrozen at step {global_step} (Stage 4) (count={unfrozen_count})")
            return True

    return False


def setup_distributed(cfg: SPITrainConfig):
    """初始化分布式训练"""
    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.world_size = dist.get_world_size()
        torch.cuda.set_device(cfg.local_rank)
        if cfg.local_rank == 0:
            print(f"🌐 Distributed training initialized: {cfg.world_size} GPUs")
    return cfg.local_rank == 0  # 返回是否为主进程


def train_spi(cfg: SPITrainConfig):
    """主训练函数"""
    is_main_process = setup_distributed(cfg)
    device = torch.device(f'cuda:{cfg.local_rank}' if cfg.distributed else cfg.device)

    # 自动检测数据集类型并选择合适的加载器
    from pathlib import Path
    data_root = Path(cfg.data_root)

    # 检查是否有multi-expert数据集
    expert_files_exist = any([
        (data_root / f"{expert}_enhanced_36.f32").exists() or
        (data_root / f"{expert}_200k_36.f32").exists()
        for expert in ["harmonic", "transient", "burst_inpaint", "low_snr"]
    ])

    if expert_files_exist:
        if is_main_process:
            print("Multi-expert数据集检测到，使用CombinedExpertDataset")
        dataloader, dataset = create_combined_data_loader(
            data_root=cfg.data_root,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size // cfg.world_size,  # 调整batch_size
            max_samples=None,
            num_workers=8 // cfg.world_size,  # 调整worker数量
            energy_selection=True,
        )
    else:
        if is_main_process:
            print("单一数据集检测到，使用AETHERRealDataset")
        # 检查基础数据文件
        features_file = str(data_root / "out_features.f32")
        audio_file = str(data_root / "out_speech.pcm")

        if not (data_root / "out_features.f32").exists():
            raise FileNotFoundError(f"未找到特征文件: {features_file}")
        if not (data_root / "out_speech.pcm").exists():
            raise FileNotFoundError(f"未找到音频文件: {audio_file}")

        dataloader, dataset = create_aether_data_loader(
            data_dir=cfg.data_root,
            sequence_length=cfg.sequence_length,
            batch_size=cfg.batch_size // cfg.world_size,  # 调整batch_size
            features_file=features_file,
            audio_file=audio_file,
            max_samples=None,
            num_workers=8 // cfg.world_size,  # 调整worker数量
            energy_selection=True,
            feature_spec_type="fargan",  # 使用FARGAN兼容的36维特征规范
            distributed=cfg.distributed  # 传递分布式标志
        )

    # 构建模型
    model = build_spi_model(cfg)
    model.to(device)
    if is_main_process:
        model.print_model_info()

    # 加载并冻结FARGAN声码器（vocoder）权重
    init_fargan_from_checkpoint(model, cfg, device)

    # 分布式模型包装
    if cfg.distributed:
        model = DDP(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank,
                   find_unused_parameters=True)
        if is_main_process:
            print(f"🔗 Model wrapped with DistributedDataParallel")

    # 信道模拟器
    channel_sim = ChannelSimulator(sample_rate=16000, frame_hz=100)

    # 增强音频损失（F0 + 感知损失）
    enhanced_loss = None
    if cfg.use_enhanced_losses:
        enhanced_loss = create_enhanced_audio_loss(
            sr=16000,
            f0_weight=cfg.f0_loss_weight,
            perceptual_weight=cfg.perceptual_loss_weight,
            use_f0_loss=True,
            use_perceptual_loss=True
        ).to(device)
        print(f"Initialized enhanced losses: F0 weight={cfg.f0_loss_weight}, Perceptual weight={cfg.perceptual_loss_weight}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    # 渐进式损失调度器 (Anti-Buzz核心组件)
    loss_scheduler = ProgressiveLossScheduler(
        stage1_steps=1000,    # Stage 1: 优先特征级监督
        stage2_steps=3000,    # Stage 2: 加入音频级监督，弱GAN
        stage3_steps=5000,    # Stage 3: 逐步恢复GAN权重
        max_adv_weight=cfg.lambda_adv,  # 使用配置中的最大对抗权重
        max_spi_weight=cfg.lambda_spi   # 使用配置中的最大SPI权重
    )

    # 对抗损失和判别器 (进一步弱化)
    adv_wave_loss = create_adversarial_wave_loss(
        fft_sizes=[256],             # 只保留一个频率尺度
        hop_factors=4,
        base_channels=4,             # 进一步减少判别器能力
        feature_match_weight=1.0,    # 降低特征匹配权重
        adversarial_weight=0.5,      # 降低对抗权重
    ).to(device)

    disc_optimizer = torch.optim.AdamW(
        adv_wave_loss.get_discriminator_parameters(),
        lr=5e-5, betas=(0.5, 0.9)    # 更低的学习率
    )

    # 恢复检查点
    start_epoch = 0
    global_step = 0
    if cfg.resume_from:
        start_epoch, global_step = load_checkpoint(
            cfg.resume_from, model, optimizer, disc_optimizer, device
        )

    # 训练循环
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()

        # 创建带进度条的dataloader
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}",
                         leave=True, dynamic_ncols=True)

        for batch in epoch_pbar:
            feats_raw = batch["x"].to(device)[..., :20]  # 只用前20维声学特征
            audio = batch["audio"].to(device)

            # 确定训练阶段
            stage = determine_training_stage(global_step, cfg)

            # 检查是否需要解冻FARGAN (根据渐进式调度器)
            if unfreeze_fargan_vocoder(model, global_step, loss_scheduler):
                # 重新创建优化器以包含新解冻的参数
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
                tqdm.write(f"Step {global_step}: FARGAN vocoder unfrozen, optimizer recreated")

            # 检查阶段切换
            current_stage_name = loss_scheduler.get_weights(global_step)['stage_name']
            prev_stage_name = loss_scheduler.get_weights(global_step - 1)['stage_name'] if global_step > 0 else ''
            if current_stage_name != prev_stage_name:
                tqdm.write(f"Step {global_step}: Training stage switched to '{current_stage_name}'")

            # === SPI前向传播 ===
            spi_output = forward_spi_progressive(
                model=model,
                feats=feats_raw,
                audio=audio,
                channel_sim=channel_sim,
                stage=stage,
                cfg=cfg,
                device=device,
            )

            # 添加原始特征到输出 (用于损失计算)
            spi_output['feats'] = feats_raw

            audio_hat = spi_output['audio_hat']
            audio_real = spi_output['audio_real']

            # === 判别器步骤 ===
            if adv_wave_loss is not None and global_step % 3 == 0:  # 每3步训练1次判别器
                disc_optimizer.zero_grad()
                try:
                    disc_out = adv_wave_loss.discriminator_step(audio_real, audio_hat.detach())
                    loss_d = disc_out["discriminator_loss"]
                    loss_d.backward()
                    disc_optimizer.step()
                except:
                    loss_d = torch.zeros(1, device=device)
            else:
                loss_d = torch.zeros(1, device=device)

            # === 生成器步骤 ===
            optimizer.zero_grad()

            total_loss, loss_dict = compute_spi_training_loss(
                spi_output, cfg, adv_wave_loss, enhanced_loss, device, model,
                global_step=global_step, loss_scheduler=loss_scheduler
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # === 更新进度条信息 ===
            # 获取当前调度器信息
            scheduler_weights = loss_scheduler.get_weights(global_step)

            # 构建进度条postfix信息
            postfix_dict = {
                'step': global_step,
                'stage': scheduler_weights['stage_name'][:8],  # 缩短显示
                'total': f"{total_loss.item():.3f}",
                'recon': f"{loss_dict['recon'].item():.3f}",
                'adv': f"{loss_dict['adv'].item():.3f}",
                'spi': f"{loss_dict.get('spi_total', torch.zeros(1)).item():.3f}",
                'fargan': 'unfreeze' if not scheduler_weights['fargan_freeze'] else 'frozen'
            }

            # 添加Anti-Buzz特定损失信息
            if 'feat_recon' in loss_dict:
                postfix_dict['feat'] = f"{loss_dict['feat_recon'].item():.3f}"
            if 'enhanced' in loss_dict and loss_dict['enhanced'].item() > 0:
                postfix_dict['f0'] = f"{loss_dict.get('enhanced_f0', torch.zeros(1)).item():.3f}"

            epoch_pbar.set_postfix(postfix_dict)

            # === 保存检查点 ===
            if global_step % cfg.save_every_steps == 0 and global_step > 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    disc_optimizer=disc_optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    output_dir=cfg.output_dir,
                )
                tqdm.write(f"Step {global_step}: Checkpoint saved to {cfg.output_dir}")

            # === 保存音频样本 ===
            if cfg.save_audio and global_step % cfg.save_audio_every_steps == 0:
                save_audio_samples(
                    audio_real=audio_real,
                    audio_gen=audio_hat,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                )
                tqdm.write(f"Step {global_step}: Audio samples saved")

            # === 保存可视化对比图 ===
            if cfg.save_visualization and global_step % cfg.save_visualization_every_steps == 0:
                try:
                    # 创建F0和Mel谱图对比
                    create_batch_comparison_plots(
                        audio_real_batch=audio_real,
                        audio_gen_batch=audio_hat,
                        save_dir=cfg.visualization_save_dir,
                        step=global_step,
                        max_samples=cfg.max_visualization_samples,
                        sr=16000
                    )

                    # 同时保存音频文件用于播放验证
                    save_comparison_audio_samples(
                        audio_real_batch=audio_real,
                        audio_gen_batch=audio_hat,
                        save_dir=cfg.visualization_save_dir,
                        step=global_step,
                        max_samples=cfg.max_visualization_samples,
                        sr=16000
                    )

                    tqdm.write(f"Step {global_step}: Visualization saved")

                except Exception as e:
                    tqdm.write(f"Step {global_step}: Failed to save visualization: {e}")

            global_step += 1

        # Epoch完成消息
        tqdm.write(f"Epoch {epoch+1} completed!")

    tqdm.write("SPI Anti-Buzz training completed!")


def parse_args() -> SPITrainConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SPI-JSCC training")

    parser.add_argument("--data_root", type=str, default="/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/data_expert_augmented_small200k")
    parser.add_argument("--stage", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--enable_hash", action="store_true", default=False, help="Enable Hash bottleneck (Step1: start with False)")
    parser.add_argument(
        "--fargan_ckpt",
        type=str,
        default="/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/fargan_pt/fargan_sq1Ab_adv_50.pth",
        help="FARGAN 预训练权重路径（将被加载到vocoder并冻结）",
    )
    parser.add_argument("--fargan_unfreeze_steps", type=int, default=15000, help="FARGAN解冻步数")
    parser.add_argument("--batch_size", type=int, default=8)  # 稍微减小批次
    parser.add_argument("--sequence_length", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 保存相关
    parser.add_argument("--output_dir", type=str, default="./checkpoints_spi_jscc")
    parser.add_argument("--save_every_steps", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)

    # 音频验证
    parser.add_argument("--save_audio", action="store_true", default=True)
    parser.add_argument("--audio_save_dir", type=str, default="./audio_samples_spi")
    parser.add_argument("--save_audio_every_steps", type=int, default=100)
    parser.add_argument("--max_audio_samples", type=int, default=4)

    # 可视化
    parser.add_argument("--save_visualization", action="store_true", default=True)
    parser.add_argument("--visualization_save_dir", type=str, default="./visualizations_spi")
    parser.add_argument("--save_visualization_every_steps", type=int, default=100)
    parser.add_argument("--max_visualization_samples", type=int, default=3)

    # SPI参数
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--semantic_dim", type=int, default=16)
    parser.add_argument("--d_z", type=int, default=32)
    parser.add_argument("--d_s", type=int, default=24)

    # 渐进式训练
    parser.add_argument("--use_progressive", action="store_true", default=True)
    parser.add_argument("--stage1_steps", type=int, default=200)
    parser.add_argument("--stage2_steps", type=int, default=400)
    parser.add_argument("--stage3_steps", type=int, default=600)

    # 损失权重 - 干净baseline默认配置
    parser.add_argument("--lambda_wave", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=0.0)
    parser.add_argument("--lambda_spi", type=float, default=0.5)
    parser.add_argument("--use_enhanced_losses", action="store_true", default=False, help="Use enhanced F0 and perceptual losses")
    parser.add_argument("--f0_loss_weight", type=float, default=1.0, help="Weight for F0 consistency loss")
    parser.add_argument("--perceptual_loss_weight", type=float, default=0.1, help="Weight for perceptual losses")

    # 其他
    parser.add_argument("--use_stft_loss", action="store_true", default=True)
    parser.add_argument("--snr_min_db", type=float, default=-5.0)
    parser.add_argument("--snr_max_db", type=float, default=5.0)

    # 多GPU支持
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed training")

    args = parser.parse_args()

    return SPITrainConfig(
        data_root=args.data_root,
        stage=args.stage,
        fargan_ckpt=args.fargan_ckpt,
        fargan_unfreeze_steps=args.fargan_unfreeze_steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        save_every_steps=args.save_every_steps,
        resume_from=args.resume_from,
        save_audio=args.save_audio,
        audio_save_dir=args.audio_save_dir,
        save_audio_every_steps=args.save_audio_every_steps,
        max_audio_samples=args.max_audio_samples,
        save_visualization=args.save_visualization,
        visualization_save_dir=args.visualization_save_dir,
        save_visualization_every_steps=args.save_visualization_every_steps,
        max_visualization_samples=args.max_visualization_samples,
        img_size=args.img_size,
        semantic_dim=args.semantic_dim,
        d_z=args.d_z,
        d_s=args.d_s,
        use_progressive=args.use_progressive,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        stage3_steps=args.stage3_steps,
        lambda_wave=args.lambda_wave,
        lambda_adv=args.lambda_adv,
        lambda_spi=args.lambda_spi,
        use_enhanced_losses=args.use_enhanced_losses,
        f0_loss_weight=args.f0_loss_weight,
        perceptual_loss_weight=args.perceptual_loss_weight,
        use_stft_loss=args.use_stft_loss,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        distributed=args.distributed,
        local_rank=args.local_rank,
        world_size=args.world_size,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_spi(cfg)
