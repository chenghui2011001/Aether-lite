#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-STFT Discriminator and Adversarial Wave Loss

从final_version stage4训练代码中提取的多尺度STFT判别器和对抗训练波形损失计算组件
用于高质量的对抗训练，提升音频生成质量。

Key Features:
- STFTSubDiscriminator: 单尺度STFT判别子网络
- WaveDiscriminator: 多尺度STFT判别器组合
- adversarial_wave_loss: 完整的对抗训练损失计算
- feature_matching_loss: 特征匹配损失
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTSubDiscriminator(nn.Module):
    """
    单尺度 STFT 判别子网络：
    - 输入为单个尺度的 STFT 幅度谱 [B, F, T]
    - 使用 2D 卷积，时间维上膨胀 (1,2,4)，频率维 stride=2
    - 输出多层特征 + 最终 score 特征图
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        layers = []

        # 初始卷积：轻微 time/freq 感受野
        layers.append(
            nn.Conv2d(in_channels, c, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 三层带时间膨胀、频率下采样的卷积
        for dilation_t in (1, 2, 4):
            layers.append(
                nn.Conv2d(
                    c,
                    c,
                    kernel_size=(3, 9),
                    stride=(2, 1),          # 频率维 stride=2
                    padding=(1, 4 * dilation_t),
                    dilation=(1, dilation_t),
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.feature_layers = nn.ModuleList(layers)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, mag: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            mag: [B, F, T] STFT 幅度

        Returns:
            [feat1, feat2, ..., score]，其中 score 为 [B, 1, F', T']
        """
        x = mag.unsqueeze(1)  # [B, 1, F, T]
        feats = []
        h = x

        for layer in self.feature_layers:
            h = layer(h)
            feats.append(h)

        score = self.out_conv(h)
        feats.append(score)
        return feats


class WaveDiscriminator(nn.Module):
    """
    多尺度 STFT 判别器（适配论文中的 multi-scale STFT discriminator 思路）。

    - 对输入波形计算多个窗口长度的 STFT 幅度谱 [1024, 512, 256]
    - 每个尺度使用一个 STFTSubDiscriminator
    - 输出格式与 adv_train_fargan.py 中判别器兼容：
      List[scale]，其中每个 scale 是 [feat1, feat2, ..., final_score]
    """

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_factors: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 512, 256]

        self.fft_sizes = list(fft_sizes)
        self.hop_factors = hop_factors
        self.sub_discriminators = nn.ModuleList(
            [STFTSubDiscriminator(in_channels=1, base_channels=base_channels) for _ in self.fft_sizes]
        )

    @staticmethod
    def _stft_mag_for_disc(x: torch.Tensor, fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """
        计算判别器用的 STFT 幅度谱，强制 float32 & log1p 幅度，输出 [B, F, T].
        """
        x32 = x.to(torch.float32)
        window = torch.hann_window(win_length, device=x32.device, dtype=torch.float32)
        spec = torch.stft(
            x32,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        mag = torch.abs(spec).clamp_min(1e-4)
        return torch.log1p(mag)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] 或 [B, T] 波形

        Returns:
            List[scale]，每个 scale 是 [feat1, feat2, ..., score]
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        assert x.dim() == 2, f"WaveDiscriminator expects [B, T] or [B,1,T], got {x.shape}"

        outputs: List[List[torch.Tensor]] = []
        for fs, sub_disc in zip(self.fft_sizes, self.sub_discriminators):
            hop = max(1, fs // self.hop_factors)
            win_len = fs
            mag = self._stft_mag_for_disc(x, fs, hop, win_len)  # [B, F, T]
            feats = sub_disc(mag)
            outputs.append(feats)
        return outputs


def feature_matching_loss(scores_real: List[List[torch.Tensor]],
                         scores_gen: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Feature matching loss，参考 dnn/torch/fargan/adv_train_fargan.py，并放大整体权重.

    Args:
        scores_real: 真实音频的判别器特征 [scale][layer]
        scores_gen: 生成音频的判别器特征 [scale][layer]

    Returns:
        feature_matching_loss: 特征匹配损失
    """
    num_discs = len(scores_real)
    loss_feat = 0.0

    for k in range(num_discs):
        num_layers = len(scores_gen[k]) - 1  # 排除最后的score层
        if num_layers <= 0:
            continue

        f = 4.0 / float(num_discs * num_layers)
        for l in range(num_layers):
            loss_feat = loss_feat + f * F.l1_loss(scores_gen[k][l], scores_real[k][l].detach())

    # 论文中建议再整体乘以 5 以放大特征匹配损失的影响
    return 5.0 * loss_feat


def discriminator_loss(scores_real: List[List[torch.Tensor]],
                      scores_gen: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    计算判别器损失（真实样本应该输出1，生成样本应该输出0）

    Args:
        scores_real: 真实音频的判别器输出 [scale][layer]，最后一层是score
        scores_gen: 生成音频的判别器输出 [scale][layer]，最后一层是score

    Returns:
        discriminator_loss: 判别器损失
    """
    loss_d = 0.0
    num_discs = len(scores_real)

    for k in range(num_discs):
        # 最后一层是判别score
        real_score = scores_real[k][-1]  # [B, 1, F', T']
        gen_score = scores_gen[k][-1]    # [B, 1, F', T']

        # 真实样本损失：希望输出接近1
        loss_real = F.mse_loss(real_score, torch.ones_like(real_score))

        # 生成样本损失：希望输出接近0
        loss_gen = F.mse_loss(gen_score, torch.zeros_like(gen_score))

        loss_d += (loss_real + loss_gen) / num_discs

    return loss_d


def generator_adversarial_loss(scores_gen: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    计算生成器对抗损失（希望判别器认为生成样本是真实的）

    Args:
        scores_gen: 生成音频的判别器输出 [scale][layer]，最后一层是score

    Returns:
        generator_adversarial_loss: 生成器对抗损失
    """
    loss_g = 0.0
    num_discs = len(scores_gen)

    for k in range(num_discs):
        # 最后一层是判别score
        gen_score = scores_gen[k][-1]  # [B, 1, F', T']

        # 生成器希望判别器输出接近1
        loss_g += F.mse_loss(gen_score, torch.ones_like(gen_score)) / num_discs

    return loss_g


class AdversarialWaveLoss(nn.Module):
    """
    完整的对抗训练波形损失计算组件

    集成了多STFT判别器、特征匹配损失、对抗损失等组件
    """

    def __init__(self,
                 fft_sizes: Optional[List[int]] = None,
                 hop_factors: int = 4,
                 base_channels: int = 32,
                 feature_match_weight: float = 10.0,
                 adversarial_weight: float = 1.0):
        super().__init__()

        self.discriminator = WaveDiscriminator(
            fft_sizes=fft_sizes,
            hop_factors=hop_factors,
            base_channels=base_channels
        )

        self.feature_match_weight = feature_match_weight
        self.adversarial_weight = adversarial_weight

    def discriminator_step(self,
                          audio_real: torch.Tensor,
                          audio_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        判别器训练步骤

        Args:
            audio_real: [B, 1, T] 或 [B, T] 真实音频
            audio_gen: [B, 1, T] 或 [B, T] 生成音频

        Returns:
            包含损失的字典
        """
        with torch.no_grad():
            scores_gen = self.discriminator(audio_gen.detach())

        scores_real = self.discriminator(audio_real)

        loss_d = discriminator_loss(scores_real, scores_gen)

        return {
            'discriminator_loss': loss_d,
            'scores_real': scores_real,
            'scores_gen': scores_gen
        }

    def generator_step(self,
                      audio_real: torch.Tensor,
                      audio_gen: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生成器训练步骤

        Args:
            audio_real: [B, 1, T] 或 [B, T] 真实音频
            audio_gen: [B, 1, T] 或 [B, T] 生成音频

        Returns:
            包含损失的字典
        """
        with torch.no_grad():
            scores_real = self.discriminator(audio_real)

        scores_gen = self.discriminator(audio_gen)

        # 特征匹配损失
        loss_fm = feature_matching_loss(scores_real, scores_gen)

        # 对抗损失
        loss_adv = generator_adversarial_loss(scores_gen)

        # 总损失
        loss_total = self.feature_match_weight * loss_fm + self.adversarial_weight * loss_adv

        return {
            'total_adversarial_loss': loss_total,
            'feature_matching_loss': loss_fm,
            'adversarial_loss': loss_adv,
            'scores_real': scores_real,
            'scores_gen': scores_gen
        }

    def get_discriminator_parameters(self):
        """获取判别器参数（用于单独的优化器）"""
        return self.discriminator.parameters()


def create_adversarial_wave_loss(fft_sizes: Optional[List[int]] = None,
                                hop_factors: int = 4,
                                base_channels: int = 32,
                                feature_match_weight: float = 10.0,
                                adversarial_weight: float = 1.0) -> AdversarialWaveLoss:
    """
    便捷工厂函数

    Args:
        fft_sizes: STFT窗口尺寸列表，默认[1024, 512, 256]
        hop_factors: hop_size = fft_size // hop_factors
        base_channels: 判别器基础通道数
        feature_match_weight: 特征匹配损失权重
        adversarial_weight: 对抗损失权重

    Returns:
        AdversarialWaveLoss实例
    """
    return AdversarialWaveLoss(
        fft_sizes=fft_sizes,
        hop_factors=hop_factors,
        base_channels=base_channels,
        feature_match_weight=feature_match_weight,
        adversarial_weight=adversarial_weight
    )


if __name__ == "__main__":
    pass