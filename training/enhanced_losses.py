#!/usr/bin/env python3
"""
增强音频损失函数：F0保持损失和感知损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class F0ConsistencyLoss(nn.Module):
    """F0保持损失 - 确保生成音频保持基频信息"""

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,
        fmin: float = 80.0,
        fmax: float = 400.0,
        loss_type: str = "mse"  # "mse", "l1", or "cosine"
    ):
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.loss_type = loss_type

    def extract_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """提取F0特征"""
        batch_size = audio.size(0)
        f0_batch = []

        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()

            try:
                f0, _, _ = librosa.pyin(
                    audio_np,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    frame_length=self.hop_length * 4,
                    fill_na=0.0
                )

                # 过滤异常值
                f0 = np.where((f0 > 0) & (f0 >= self.fmin) & (f0 <= self.fmax), f0, 0.0)

            except Exception:
                # 失败时返回零数组
                n_frames = 1 + (len(audio_np) - self.hop_length * 4) // self.hop_length
                f0 = np.zeros(max(1, n_frames))

            f0_batch.append(torch.tensor(f0, dtype=torch.float32))

        # 对齐长度并堆叠
        min_len = min(f0.size(0) for f0 in f0_batch)
        f0_aligned = torch.stack([f0[:min_len] for f0 in f0_batch])

        return f0_aligned.to(audio.device)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算F0保持损失

        Args:
            y_pred: 生成音频 [B, T]
            y_true: 真实音频 [B, T]

        Returns:
            F0保持损失
        """
        # 提取F0
        f0_pred = self.extract_f0(y_pred)
        f0_true = self.extract_f0(y_true)

        # 计算有效帧掩码（真实音频有F0的帧）
        valid_mask = f0_true > 0

        if valid_mask.sum() == 0:
            # 如果没有有效F0帧，返回零损失
            return torch.tensor(0.0, device=y_pred.device)

        # 只在有效帧上计算损失
        f0_pred_valid = f0_pred[valid_mask]
        f0_true_valid = f0_true[valid_mask]

        if self.loss_type == "mse":
            # 归一化F0损失：除以典型F0值的平方以稳定训练
            f0_normalization = (self.fmax) ** 2  # ~160000 for fmax=400
            loss = F.mse_loss(f0_pred_valid, f0_true_valid) / f0_normalization
        elif self.loss_type == "l1":
            # 归一化F0损失：除以典型F0值
            f0_normalization = self.fmax  # ~400 for fmax=400
            loss = F.l1_loss(f0_pred_valid, f0_true_valid) / f0_normalization
        elif self.loss_type == "cosine":
            # 余弦相似度损失（适用于F0轮廓）
            cos_sim = F.cosine_similarity(
                f0_pred_valid.unsqueeze(0),
                f0_true_valid.unsqueeze(0),
                dim=1
            )
            loss = 1.0 - cos_sim.mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class PerceptualAudioLoss(nn.Module):
    """感知音频损失 - 基于梅尔频谱和时频域特征的感知相关损失"""

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        mel_weight: float = 1.0,
        spectral_weight: float = 0.5,
        temporal_weight: float = 0.3
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_weight = mel_weight
        self.spectral_weight = spectral_weight
        self.temporal_weight = temporal_weight

        # 预计算Mel滤波器组
        mel_fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=0, fmax=sr//2
        )
        self.register_buffer('mel_fb', torch.tensor(mel_fb, dtype=torch.float32))

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """计算对数梅尔频谱"""
        # STFT
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=True
        )

        # 幅度谱
        magnitude = torch.abs(stft)  # [B, F, T]

        # 应用Mel滤波器
        mel_spec = torch.matmul(self.mel_fb.to(audio.device), magnitude)

        # 转换为对数域
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))

        return log_mel

    def compute_spectral_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算频谱特征"""
        # STFT
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=True
        )

        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # 频谱质心（Spectral Centroid）
        freqs = torch.linspace(0, self.sr//2, magnitude.size(1)).to(audio.device)
        freqs = freqs.view(1, -1, 1)

        spec_centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)

        # 频谱滚降（Spectral Rolloff）
        magnitude_cumsum = torch.cumsum(magnitude, dim=1)
        total_magnitude = magnitude_cumsum[:, -1:, :]
        rolloff_threshold = 0.85 * total_magnitude

        # 找到85%能量点
        rolloff_indices = torch.argmax((magnitude_cumsum >= rolloff_threshold).float(), dim=1)
        spec_rolloff = freqs.squeeze()[rolloff_indices]

        return {
            'centroid': spec_centroid,
            'rolloff': spec_rolloff,
            'magnitude': magnitude
        }

    def compute_temporal_features(self, audio: torch.Tensor) -> torch.Tensor:
        """计算时域特征（包络等）"""
        # 音频包络（通过希尔伯特变换或简单滑动平均）
        # 使用滑动窗口RMS作为包络近似
        window_size = self.hop_length
        audio_padded = F.pad(audio, (window_size//2, window_size//2), mode='reflect')

        envelope = []
        for i in range(0, audio.size(-1), window_size//4):  # 4x重叠
            if i + window_size <= audio_padded.size(-1):
                window = audio_padded[..., i:i+window_size]
                rms = torch.sqrt(torch.mean(window**2, dim=-1, keepdim=True))
                envelope.append(rms)

        if envelope:
            envelope = torch.cat(envelope, dim=-1)
        else:
            envelope = torch.zeros(audio.size(0), 1).to(audio.device)

        return envelope

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算感知音频损失

        Args:
            y_pred: 生成音频 [B, T]
            y_true: 真实音频 [B, T]

        Returns:
            包含各组件损失的字典
        """
        losses = {}

        # 对齐长度
        min_len = min(y_pred.size(-1), y_true.size(-1))
        y_pred = y_pred[..., :min_len]
        y_true = y_true[..., :min_len]

        # 1. Mel频谱损失 (归一化处理)
        mel_pred = self.compute_mel_spectrogram(y_pred)
        mel_true = self.compute_mel_spectrogram(y_true)
        # 归一化：典型对数mel值范围约为10 (从-10到0)
        mel_normalization = 10.0
        losses['mel'] = F.l1_loss(mel_pred, mel_true) / mel_normalization * self.mel_weight

        # 2. 频谱特征损失
        if self.spectral_weight > 0:
            spec_pred = self.compute_spectral_features(y_pred)
            spec_true = self.compute_spectral_features(y_true)

            # 频谱质心损失 (归一化到采样率一半)
            centroid_normalization = self.sr / 2  # ~8000 for 16kHz
            centroid_loss = F.l1_loss(spec_pred['centroid'], spec_true['centroid']) / centroid_normalization

            # 频谱滚降损失 (同样归一化)
            rolloff_loss = F.l1_loss(spec_pred['rolloff'], spec_true['rolloff']) / centroid_normalization

            losses['spectral'] = (centroid_loss + rolloff_loss) * self.spectral_weight

        # 3. 时域包络损失
        if self.temporal_weight > 0:
            env_pred = self.compute_temporal_features(y_pred)
            env_true = self.compute_temporal_features(y_true)

            # 对齐长度
            min_env_len = min(env_pred.size(-1), env_true.size(-1))
            env_pred = env_pred[..., :min_env_len]
            env_true = env_true[..., :min_env_len]

            losses['temporal'] = F.l1_loss(env_pred, env_true) * self.temporal_weight

        # 总感知损失
        losses['total'] = sum(losses.values())

        return losses


class EnhancedAudioLoss(nn.Module):
    """组合增强音频损失"""

    def __init__(
        self,
        sr: int = 16000,
        use_f0_loss: bool = True,
        use_perceptual_loss: bool = True,
        f0_weight: float = 10.0,  # 降低默认权重，配合归一化
        perceptual_weight: float = 0.1,  # 降低感知损失默认权重
        **kwargs
    ):
        super().__init__()
        self.use_f0_loss = use_f0_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.f0_weight = f0_weight
        self.perceptual_weight = perceptual_weight

        if use_f0_loss:
            self.f0_loss = F0ConsistencyLoss(sr=sr, **kwargs.get('f0_config', {}))

        if use_perceptual_loss:
            self.perceptual_loss = PerceptualAudioLoss(sr=sr, **kwargs.get('perceptual_config', {}))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算组合增强损失

        Returns:
            包含所有损失组件的字典
        """
        losses = {}

        if self.use_f0_loss:
            losses['f0'] = self.f0_loss(y_pred, y_true) * self.f0_weight

        if self.use_perceptual_loss:
            perceptual_losses = self.perceptual_loss(y_pred, y_true)
            for key, value in perceptual_losses.items():
                losses[f'perceptual_{key}'] = value * self.perceptual_weight

        # 计算总的增强损失
        losses['enhanced_total'] = sum(losses.values())

        return losses


# 便捷函数
def create_enhanced_audio_loss(
    sr: int = 16000,
    f0_weight: float = 10.0,   # 配合归一化的新默认权重
    perceptual_weight: float = 0.1,  # 降低感知损失权重
    **kwargs
) -> EnhancedAudioLoss:
    """创建增强音频损失函数"""
    return EnhancedAudioLoss(
        sr=sr,
        f0_weight=f0_weight,
        perceptual_weight=perceptual_weight,
        **kwargs
    )