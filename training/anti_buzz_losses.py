#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anti-Buzz损失函数：专门打击F0/Voicing作弊解

核心思想：让"F0塌缩+全程有声"的作弊行为在损失函数中变得极其昂贵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, Tuple, Optional


def compute_f0_correlation(pred_f0: torch.Tensor, target_f0: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算F0轨迹的相关性，防止只拟合均值

    Args:
        pred_f0: [B, T] 预测F0
        target_f0: [B, T] 目标F0
        eps: 数值稳定性

    Returns:
        相关性系数 [B]
    """
    B, T = pred_f0.shape

    # 去中心化
    pred_centered = pred_f0 - pred_f0.mean(dim=1, keepdim=True)
    target_centered = target_f0 - target_f0.mean(dim=1, keepdim=True)

    # 计算相关性
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = pred_centered.std(dim=1)
    target_std = target_centered.std(dim=1)

    correlation = numerator / (pred_std * target_std + eps)
    return correlation


def extract_f0_simple(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160) -> torch.Tensor:
    """
    简单的F0提取器（基于autocorrelation）

    Args:
        audio: [B, L] 音频信号
        sr: 采样率
        hop_length: 帧移

    Returns:
        f0: [B, T] F0轨迹
    """
    B, L = audio.shape
    n_frames = L // hop_length

    # 简化版本：使用能量作为F0代理（实际应用中建议用专业F0提取器）
    f0_frames = []

    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length * 2, L)  # 重叠窗口
        frame = audio[:, start:end]  # [B, frame_size]

        # 计算自相关峰值作为F0估计
        if frame.size(1) > 0:
            # 简单版：用主频率成分估计
            fft = torch.fft.fft(frame, dim=1)
            magnitude = torch.abs(fft)

            # 找到50-400Hz范围内的峰值
            freq_bins = torch.fft.fftfreq(frame.size(1), 1/sr)
            valid_mask = (freq_bins >= 50) & (freq_bins <= 400)

            if valid_mask.sum() > 0:
                valid_magnitude = magnitude[:, valid_mask]
                peak_indices = torch.argmax(valid_magnitude, dim=1)
                valid_freqs = freq_bins[valid_mask]
                f0_frame = valid_freqs[peak_indices]
            else:
                f0_frame = torch.zeros(B, device=audio.device)
        else:
            f0_frame = torch.zeros(B, device=audio.device)

        f0_frames.append(f0_frame)

    f0 = torch.stack(f0_frames, dim=1)  # [B, T]
    return f0


def compute_energy_mask(audio: torch.Tensor, threshold: float = 0.01, hop_length: int = 160) -> torch.Tensor:
    """
    计算静音掩码

    Args:
        audio: [B, L] 音频信号
        threshold: 静音阈值
        hop_length: 帧移

    Returns:
        silence_mask: [B, T] 静音掩码（1=静音，0=有声）
    """
    B, L = audio.shape
    n_frames = L // hop_length

    energy_frames = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + hop_length, L)
        frame = audio[:, start:end]

        # RMS能量
        energy = torch.sqrt(torch.mean(frame ** 2, dim=1))
        energy_frames.append(energy)

    energy = torch.stack(energy_frames, dim=1)  # [B, T]
    silence_mask = (energy < threshold).float()

    return silence_mask


class AntiBuzzLoss(nn.Module):
    """
    Anti-Buzz损失：专门打击F0/Voicing作弊解

    核心策略：
    1. F0/Voicing维度加权监督
    2. F0形状相关性约束
    3. Voicing二分类损失
    4. 音频级F0监督
    5. F0方差下界约束
    6. 静音段能量惩罚
    """

    def __init__(
        self,
        f0_weight: float = 5.0,
        vuv_weight: float = 5.0,
        f0_shape_weight: float = 1.0,
        vuv_bce_weight: float = 2.0,
        audio_f0_weight: float = 3.0,
        f0_variance_weight: float = 2.0,
        silence_weight: float = 3.0,
        min_f0_std: float = 20.0,
        silence_threshold: float = 0.01,
        sr: int = 16000,
        hop_length: int = 160
    ):
        super().__init__()

        self.f0_weight = f0_weight
        self.vuv_weight = vuv_weight
        self.f0_shape_weight = f0_shape_weight
        self.vuv_bce_weight = vuv_bce_weight
        self.audio_f0_weight = audio_f0_weight
        self.f0_variance_weight = f0_variance_weight
        self.silence_weight = silence_weight
        self.min_f0_std = min_f0_std
        self.silence_threshold = silence_threshold
        self.sr = sr
        self.hop_length = hop_length

    def forward(
        self,
        feats_pred: torch.Tensor,
        feats_target: torch.Tensor,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算Anti-Buzz损失

        Args:
            feats_pred: [B, T, 20] 预测特征
            feats_target: [B, T, 20] 目标特征
            audio_pred: [B, L] 预测音频
            audio_target: [B, L] 目标音频

        Returns:
            total_loss, loss_dict
        """
        B, T, D = feats_pred.shape
        losses = {}

        # 确保特征维度正确
        if D < 20:
            # 如果维度不足，用零填充
            padding = torch.zeros(B, T, 20 - D, device=feats_pred.device, dtype=feats_pred.dtype)
            feats_pred = torch.cat([feats_pred, padding], dim=-1)
            feats_target = torch.cat([feats_target, padding], dim=-1)

        # ===== 特征级监督 =====

        # 1. 基础特征重建（前18维：倒谱等）
        cep_diff = feats_pred[..., :18] - feats_target[..., :18]
        losses['cep_recon'] = cep_diff.abs().mean()

        # 2. F0维度加权监督（第19维）
        f0_diff = feats_pred[..., 18:19] - feats_target[..., 18:19]
        losses['f0_feat'] = f0_diff.abs().mean()

        # 3. Voicing维度加权监督（第20维）
        vuv_diff = feats_pred[..., 19:20] - feats_target[..., 19:20]
        losses['vuv_feat'] = vuv_diff.abs().mean()

        # 4. F0形状相关性约束
        f0_pred_1d = feats_pred[..., 18]  # [B, T]
        f0_target_1d = feats_target[..., 18]  # [B, T]

        f0_correlation = compute_f0_correlation(f0_pred_1d, f0_target_1d)
        losses['f0_shape'] = (1.0 - f0_correlation).clamp(min=0).mean()

        # 5. Voicing二分类损失 (使用logits版本，混合精度安全)
        vuv_target = (feats_target[..., 19] > 0.5).float()  # 转为0/1标签
        vuv_pred_logits = feats_pred[..., 19]  # 直接使用logits
        losses['vuv_bce'] = F.binary_cross_entropy_with_logits(vuv_pred_logits, vuv_target)

        # ===== 音频级监督 =====

        try:
            # 6. 音频级F0损失
            with torch.no_grad():
                f0_audio_target = extract_f0_simple(audio_target, self.sr, self.hop_length)

            f0_audio_pred = extract_f0_simple(audio_pred, self.sr, self.hop_length)

            # 对齐长度
            min_len = min(f0_audio_target.size(1), f0_audio_pred.size(1))
            f0_audio_target = f0_audio_target[:, :min_len]
            f0_audio_pred = f0_audio_pred[:, :min_len]

            losses['f0_audio'] = F.l1_loss(f0_audio_pred, f0_audio_target)

            # 7. F0方差下界约束（防止塌缩为常数）
            f0_std_pred = f0_audio_pred.std(dim=1)  # [B]
            losses['f0_variance'] = F.relu(self.min_f0_std - f0_std_pred).mean()

        except Exception as e:
            # 如果F0提取失败，使用零损失
            print(f"Warning: F0 extraction failed: {e}")
            losses['f0_audio'] = torch.zeros(1, device=feats_pred.device)
            losses['f0_variance'] = torch.zeros(1, device=feats_pred.device)

        try:
            # 8. 静音段能量惩罚
            silence_mask = compute_energy_mask(audio_target, self.silence_threshold, self.hop_length)

            # 计算音频差异
            min_audio_len = min(audio_pred.size(1), audio_target.size(1))
            audio_pred_aligned = audio_pred[:, :min_audio_len]
            audio_target_aligned = audio_target[:, :min_audio_len]
            audio_diff = (audio_pred_aligned - audio_target_aligned).abs()

            # 在静音段应用加权惩罚
            if silence_mask.size(1) > 0:
                # 上采样silence_mask到音频长度
                silence_mask_audio = F.interpolate(
                    silence_mask.unsqueeze(1),
                    size=min_audio_len,
                    mode='nearest'
                ).squeeze(1)  # [B, L]

                losses['silence_penalty'] = (audio_diff * silence_mask_audio).mean()
            else:
                losses['silence_penalty'] = torch.zeros(1, device=feats_pred.device)

        except Exception as e:
            print(f"Warning: Silence penalty computation failed: {e}")
            losses['silence_penalty'] = torch.zeros(1, device=feats_pred.device)

        # ===== 总损失 =====

        total_loss = (
            losses['cep_recon'] +  # 基础重建：权重1.0
            self.f0_weight * losses['f0_feat'] +  # F0特征：权重5.0
            self.vuv_weight * losses['vuv_feat'] +  # VUV特征：权重5.0
            self.f0_shape_weight * losses['f0_shape'] +  # F0形状：权重1.0
            self.vuv_bce_weight * losses['vuv_bce'] +  # VUV分类：权重2.0
            self.audio_f0_weight * losses['f0_audio'] +  # 音频F0：权重3.0
            self.f0_variance_weight * losses['f0_variance'] +  # F0方差：权重2.0
            self.silence_weight * losses['silence_penalty']  # 静音惩罚：权重3.0
        )

        losses['total'] = total_loss

        return total_loss, losses


class ProgressiveLossScheduler:
    """
    渐进式损失权重调度器

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
            'fargan_freeze': True
        }

        if global_step < self.stage1_steps:
            # Stage 1: 优先特征级监督
            weights['lambda_adv'] = 0.0
            weights['lambda_spi'] = self.max_spi_weight
            weights['fargan_freeze'] = True

        elif global_step < self.stage2_steps:
            # Stage 2: 加入音频级监督，弱GAN
            weights['lambda_adv'] = 0.1 * self.max_adv_weight
            weights['lambda_spi'] = self.max_spi_weight
            weights['fargan_freeze'] = True

        elif global_step < self.stage3_steps:
            # Stage 3: 逐步恢复GAN权重
            progress = (global_step - self.stage2_steps) / (self.stage3_steps - self.stage2_steps)
            weights['lambda_adv'] = progress * self.max_adv_weight
            weights['lambda_spi'] = self.max_spi_weight
            weights['fargan_freeze'] = True

        else:
            # Stage 4: 解冻FARGAN，端到端优化
            weights['lambda_adv'] = self.max_adv_weight
            weights['lambda_spi'] = self.max_spi_weight * 0.8  # 略微降低SPI权重
            weights['fargan_freeze'] = False

        return weights


def test_anti_buzz_loss():
    """测试Anti-Buzz损失函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试数据
    B, T, L = 2, 100, 16000
    feats_pred = torch.randn(B, T, 20, device=device)
    feats_target = torch.randn(B, T, 20, device=device)
    audio_pred = torch.randn(B, L, device=device) * 0.1
    audio_target = torch.randn(B, L, device=device) * 0.1

    # 创建损失函数
    loss_fn = AntiBuzzLoss().to(device)

    # 计算损失
    total_loss, loss_dict = loss_fn(feats_pred, feats_target, audio_pred, audio_target)

    print("Anti-Buzz Loss Test Results:")
    print(f"Total Loss: {total_loss.item():.6f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.6f}")

    # 测试调度器
    scheduler = ProgressiveLossScheduler()
    for step in [0, 500, 1500, 3500, 6000]:
        weights = scheduler.get_weights(step)
        print(f"Step {step}: {weights}")

    print("✓ Anti-Buzz Loss test completed!")


if __name__ == "__main__":
    test_anti_buzz_loss()