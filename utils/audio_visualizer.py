#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频可视化工具：生成F0和Mel谱图对比图
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import os
from typing import Tuple, Optional
import torchaudio
import warnings
warnings.filterwarnings('ignore')


def extract_f0(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160) -> np.ndarray:
    """提取F0（基频）"""
    audio_np = audio.detach().cpu().numpy()

    # 使用librosa提取F0，限制到合理的人声范围
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=80,    # 人声最低频率 80Hz
            fmax=400,   # 人声最高频率 400Hz (降低上限避免异常高频)
            sr=sr,
            hop_length=hop_length,
            frame_length=hop_length * 4,
            fill_na=0.0
        )

        # 过滤异常值：超出合理范围的F0设为0
        f0 = np.where((f0 > 0) & (f0 >= 80) & (f0 <= 400), f0, 0.0)

        return f0

    except Exception as e:
        print(f"F0 extraction failed: {e}")
        # 返回全零数组，确保长度正确
        n_frames = 1 + (len(audio_np) - hop_length * 4) // hop_length
        return np.zeros(n_frames)


def extract_mel_spectrogram(audio: torch.Tensor, sr: int = 16000,
                          n_fft: int = 1024, hop_length: int = 160,
                          n_mels: int = 80) -> np.ndarray:
    """提取Mel谱图"""
    audio_np = audio.detach().cpu().numpy()

    # 计算Mel谱图
    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=sr // 2
    )

    # 转换为dB
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def create_audio_comparison_plot(
    audio_real: torch.Tensor,
    audio_gen: torch.Tensor,
    save_path: str,
    sr: int = 16000,
    title: str = "Audio Comparison",
    show_waveform: bool = True,
    hop_length: int = 160
) -> None:
    """
    创建音频对比图，包含波形、F0和Mel谱图

    Args:
        audio_real: 真实音频 [L]
        audio_gen: 生成音频 [L]
        save_path: 保存路径
        sr: 采样率
        title: 图标题
        show_waveform: 是否显示波形
    """
    # 确保Tensor已detach并移到CPU
    audio_real = audio_real.detach().cpu()
    audio_gen = audio_gen.detach().cpu()

    # 确保音频长度一致
    min_len = min(audio_real.size(0), audio_gen.size(0))
    audio_real = audio_real[:min_len]
    audio_gen = audio_gen[:min_len]

    # 归一化
    audio_real = audio_real / (audio_real.abs().max() + 1e-8)
    audio_gen = audio_gen / (audio_gen.abs().max() + 1e-8)

    # 提取特征
    f0_real = extract_f0(audio_real, sr, hop_length)
    f0_gen = extract_f0(audio_gen, sr, hop_length)

    mel_real = extract_mel_spectrogram(audio_real, sr, hop_length=hop_length)
    mel_gen = extract_mel_spectrogram(audio_gen, sr, hop_length=hop_length)

    # 创建图形
    if show_waveform:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 确保F0长度一致，使用较短的长度
    min_f0_len = min(len(f0_real), len(f0_gen))
    f0_real = f0_real[:min_f0_len]
    f0_gen = f0_gen[:min_f0_len]

    # 确保Mel长度一致
    min_mel_frames = min(mel_real.shape[1], mel_gen.shape[1])
    mel_real = mel_real[:, :min_mel_frames]
    mel_gen = mel_gen[:, :min_mel_frames]

    # 时间轴计算
    time_audio = np.arange(audio_real.size(0)) / sr
    time_f0 = np.arange(min_f0_len) * hop_length / sr if min_f0_len > 0 else np.array([0])
    time_mel = np.arange(min_mel_frames) * hop_length / sr if min_mel_frames > 0 else np.array([0])

    row_idx = 0

    # 1. 波形对比（如果启用）
    if show_waveform:
        axes[row_idx, 0].plot(time_audio, audio_real.numpy(), 'b-', alpha=0.7, linewidth=0.5)
        axes[row_idx, 0].set_title('Real Audio Waveform', fontweight='bold')
        axes[row_idx, 0].set_xlabel('Time (s)')
        axes[row_idx, 0].set_ylabel('Amplitude')
        axes[row_idx, 0].grid(True, alpha=0.3)
        axes[row_idx, 0].set_ylim(-1.1, 1.1)

        axes[row_idx, 1].plot(time_audio, audio_gen.numpy(), 'r-', alpha=0.7, linewidth=0.5)
        axes[row_idx, 1].set_title('Generated Audio Waveform', fontweight='bold')
        axes[row_idx, 1].set_xlabel('Time (s)')
        axes[row_idx, 1].set_ylabel('Amplitude')
        axes[row_idx, 1].grid(True, alpha=0.3)
        axes[row_idx, 1].set_ylim(-1.1, 1.1)

        row_idx += 1

    # 2. F0对比
    # 过滤掉0值（无声段）
    f0_real_plot = f0_real.copy()
    f0_gen_plot = f0_gen.copy()
    f0_real_plot[f0_real_plot == 0] = np.nan
    f0_gen_plot[f0_gen_plot == 0] = np.nan

    # 计算F0统计信息
    f0_real_valid = f0_real_plot[~np.isnan(f0_real_plot)]
    f0_gen_valid = f0_gen_plot[~np.isnan(f0_gen_plot)]

    f0_real_mean = np.mean(f0_real_valid) if len(f0_real_valid) > 0 else 0
    f0_gen_mean = np.mean(f0_gen_valid) if len(f0_gen_valid) > 0 else 0
    f0_real_std = np.std(f0_real_valid) if len(f0_real_valid) > 0 else 0
    f0_gen_std = np.std(f0_gen_valid) if len(f0_gen_valid) > 0 else 0

    # 添加有效帧统计
    f0_real_ratio = len(f0_real_valid) / len(f0_real_plot) if len(f0_real_plot) > 0 else 0
    f0_gen_ratio = len(f0_gen_valid) / len(f0_gen_plot) if len(f0_gen_plot) > 0 else 0

    axes[row_idx, 0].plot(time_f0, f0_real_plot, 'b-', linewidth=2, label='Real F0')
    axes[row_idx, 0].set_title(f'Real Audio F0 ({f0_real_ratio:.1%} voiced)\nMean: {f0_real_mean:.1f}Hz, Std: {f0_real_std:.1f}Hz', fontweight='bold')
    axes[row_idx, 0].set_xlabel('Time (s)')
    axes[row_idx, 0].set_ylabel('F0 (Hz)')
    axes[row_idx, 0].grid(True, alpha=0.3)
    axes[row_idx, 0].set_ylim(50, 500)

    axes[row_idx, 1].plot(time_f0, f0_gen_plot, 'r-', linewidth=2, label='Generated F0')
    axes[row_idx, 1].set_title(f'Generated Audio F0 ({f0_gen_ratio:.1%} voiced)\nMean: {f0_gen_mean:.1f}Hz, Std: {f0_gen_std:.1f}Hz', fontweight='bold')
    axes[row_idx, 1].set_xlabel('Time (s)')
    axes[row_idx, 1].set_ylabel('F0 (Hz)')
    axes[row_idx, 1].grid(True, alpha=0.3)
    axes[row_idx, 1].set_ylim(50, 500)

    row_idx += 1

    # 3. Mel谱图对比
    im1 = axes[row_idx, 0].imshow(mel_real, aspect='auto', origin='lower',
                                  extent=[0, time_mel[-1], 0, mel_real.shape[0]],
                                  cmap='viridis', vmin=-80, vmax=0)
    axes[row_idx, 0].set_title('Real Audio Mel Spectrogram', fontweight='bold')
    axes[row_idx, 0].set_xlabel('Time (s)')
    axes[row_idx, 0].set_ylabel('Mel Bin')

    im2 = axes[row_idx, 1].imshow(mel_gen, aspect='auto', origin='lower',
                                  extent=[0, time_mel[-1], 0, mel_gen.shape[0]],
                                  cmap='viridis', vmin=-80, vmax=0)
    axes[row_idx, 1].set_title('Generated Audio Mel Spectrogram', fontweight='bold')
    axes[row_idx, 1].set_xlabel('Time (s)')
    axes[row_idx, 1].set_ylabel('Mel Bin')

    # 添加颜色条
    plt.colorbar(im1, ax=axes[row_idx, 0], format='%+2.0f dB', shrink=0.8)
    plt.colorbar(im2, ax=axes[row_idx, 1], format='%+2.0f dB', shrink=0.8)

    # 计算并显示统计信息
    f0_mse = np.nanmean((f0_real[:len(f0_gen)] - f0_gen)**2)
    mel_mse = np.mean((mel_real - mel_gen)**2)

    # 在图上添加统计信息
    stats_text = f'F0 MSE: {f0_mse:.2f} Hz²\nMel MSE: {mel_mse:.3f} dB²'
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Audio comparison plot saved: {save_path}")
    print(f"F0 MSE: {f0_mse:.2f} Hz², Mel MSE: {mel_mse:.3f} dB²")


def create_batch_comparison_plots(
    audio_real_batch: torch.Tensor,
    audio_gen_batch: torch.Tensor,
    save_dir: str,
    step: int,
    max_samples: int = 3,
    sr: int = 16000
) -> None:
    """
    为batch中的音频样本创建对比图

    Args:
        audio_real_batch: 真实音频批 [B, L]
        audio_gen_batch: 生成音频批 [B, L]
        save_dir: 保存目录
        step: 训练步数
        max_samples: 最大生成样本数
        sr: 采样率
    """
    batch_size = min(audio_real_batch.size(0), max_samples)

    for i in range(batch_size):
        audio_real = audio_real_batch[i]  # [L]
        audio_gen = audio_gen_batch[i]    # [L]

        save_path = os.path.join(save_dir, f"comparison_step_{step:06d}_sample_{i:02d}.png")
        title = f"Audio Comparison - Step {step} - Sample {i}"

        try:
            create_audio_comparison_plot(
                audio_real=audio_real,
                audio_gen=audio_gen,
                save_path=save_path,
                sr=sr,
                title=title,
                show_waveform=True,
                hop_length=160
            )
        except Exception as e:
            print(f"Failed to create comparison plot for sample {i}: {e}")


def save_comparison_audio_samples(
    audio_real_batch: torch.Tensor,
    audio_gen_batch: torch.Tensor,
    save_dir: str,
    step: int,
    max_samples: int = 3,
    sr: int = 16000
) -> None:
    """
    保存音频样本文件（配合可视化使用）

    Args:
        audio_real_batch: 真实音频批 [B, L]
        audio_gen_batch: 生成音频批 [B, L]
        save_dir: 保存目录
        step: 训练步数
        max_samples: 最大保存样本数
        sr: 采样率
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = min(audio_real_batch.size(0), max_samples)

    for i in range(batch_size):
        audio_real = audio_real_batch[i].detach().cpu()  # [L]
        audio_gen = audio_gen_batch[i].detach().cpu()    # [L]

        # 归一化
        audio_real = audio_real / (audio_real.abs().max() + 1e-8)
        audio_gen = audio_gen / (audio_gen.abs().max() + 1e-8)

        # 保存音频文件
        real_path = os.path.join(save_dir, f"step_{step:06d}_sample_{i:02d}_real.wav")
        gen_path = os.path.join(save_dir, f"step_{step:06d}_sample_{i:02d}_gen.wav")

        try:
            torchaudio.save(real_path, audio_real.unsqueeze(0), sr)
            torchaudio.save(gen_path, audio_gen.unsqueeze(0), sr)
        except Exception as e:
            print(f"Failed to save audio sample {i}: {e}")