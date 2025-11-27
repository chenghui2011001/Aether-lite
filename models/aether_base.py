#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aether-Base: 中杯老师模型

基于用户技术方案的三层Teacher-Student架构中的"中杯老师"：
- 参数量: 5-10M，比Lite大但比超大老师小
- 与Aether-Lite保持相同的轻量架构设计，只是容量更大
- 负责向Aether-Lite提供中间层蒸馏目标
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import math

# 使用本地复制的成熟组件
from .hash_bottleneck import HashBottleneck
from .fargan_decoder import FARGANDecoder

# 复用Lite版本的信道模拟函数
try:
    from .lite_speech_jscc import channel_awgn, channel_rayleigh, apply_bit_noise
except ImportError:
    from lite_speech_jscc import channel_awgn, channel_rayleigh, apply_bit_noise


class EncoderBase(nn.Module):
    """中杯编码器：特征+CSI → latent（比Lite容量更大）"""

    def __init__(self, feat_dim=36, d_csi=3, d_z=16, hidden=128, conv_ch=96):
        super().__init__()
        in_dim = feat_dim + d_csi

        # Conv1d局部特征提取（更多通道）
        self.conv1 = nn.Conv1d(in_dim, conv_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)  # 额外一层

        # GRU时序建模（更大hidden）
        self.gru = nn.GRU(
            input_size=conv_ch,
            hidden_size=hidden,
            num_layers=2,  # 保持2层，避免过度复杂
            batch_first=True,
            bidirectional=False,
        )

        # 输出投影（更深的网络）
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Linear(hidden//2, d_z),
        )

        self.norm = nn.LayerNorm(d_z)

    def forward(self, x_feat, csi):
        B, T, Fdim = x_feat.shape
        csi_t = csi.unsqueeze(1).expand(B, T, -1)  # [B,T,d_csi]
        x = torch.cat([x_feat, csi_t], dim=-1)     # [B,T,F+d_csi]

        # Conv1d特征提取（3层）
        x = x.transpose(1, 2)      # [B,C,T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.transpose(1, 2)      # [B,T,conv_ch]

        # GRU时序建模
        h, _ = self.gru(x)         # [B,T,hidden]

        # 输出投影
        z = self.proj(h)           # [B,T,d_z]
        z = self.norm(z)
        z = torch.tanh(z)          # 限幅到 [-1,1]
        return z


class JSCCEncoderBase(nn.Module):
    """中杯JSCC编码器：CSI感知的模拟码（容量更大）"""

    def __init__(self, d_z=16, d_s=16, d_csi=3, hidden=64):
        super().__init__()
        # 更深的投影网络
        self.z_proj = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_s)
        )

        # 更复杂的CSI处理
        self.csi_mlp = nn.Sequential(
            nn.Linear(d_csi, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_s),
        )

    def forward(self, z, csi):
        B, T, _ = z.shape
        h = self.z_proj(z)  # [B,T,d_s]

        # CSI感知的scale/bias
        c = self.csi_mlp(csi)        # [B,2*d_s]
        scale, bias = c.chunk(2, dim=-1)  # [B,d_s]
        scale = torch.tanh(scale).unsqueeze(1)  # [B,1,d_s]
        bias = bias.unsqueeze(1)

        # 调制
        s = (h + bias) * scale       # [B,T,d_s]

        # 功率归一化
        power = (s ** 2).mean(dim=(1, 2), keepdim=True) + 1e-6
        s = s / power.sqrt()
        return s


class JSCCDecoderBase(nn.Module):
    """中杯JSCC解码器：CSI感知的去噪器（容量更大）"""

    def __init__(self, d_z=16, d_s=16, d_csi=3, hidden=64):
        super().__init__()
        # 更复杂的CSI处理
        self.csi_mlp = nn.Sequential(
            nn.Linear(d_csi, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_s),
        )

        # 更深的投影网络
        self.s_proj = nn.Sequential(
            nn.Linear(d_s, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_z)
        )

    def forward(self, y, csi, h_rayleigh=None):
        # CSI感知的shrinkage
        gamma = torch.tanh(self.csi_mlp(csi)).unsqueeze(1)  # [B,1,d_s]
        y_d = gamma * y

        # Rayleigh均衡
        if h_rayleigh is not None:
            y_d = y_d / (h_rayleigh + 1e-3)

        z_hat = self.s_proj(y_d)
        return z_hat


class DecoderBase(nn.Module):
    """中杯解码器：latent+CSI → 特征（比Lite容量更大）"""

    def __init__(self, feat_dim=36, d_csi=3, d_z=16, hidden=128, conv_ch=96):
        super().__init__()
        in_dim = d_z + d_csi

        # GRU时序建模（更大hidden）
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # Conv1d特征重建（3层）
        self.conv1 = nn.Conv1d(hidden, conv_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(conv_ch, conv_ch, kernel_size=3, padding=1)

        # 输出投影（更深的网络）
        self.proj = nn.Sequential(
            nn.Linear(conv_ch, conv_ch),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(conv_ch, conv_ch//2),
            nn.GELU(),
            nn.Linear(conv_ch//2, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, z_hat, csi):
        B, T, _ = z_hat.shape
        csi_t = csi.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([z_hat, csi_t], dim=-1)   # [B,T,d_z+d_csi]

        # GRU时序建模
        h, _ = self.gru(x)                      # [B,T,hidden]

        # Conv1d特征重建（3层）
        h = h.transpose(1, 2)                   # [B,hidden,T]
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = h.transpose(1, 2)                   # [B,T,conv_ch]

        # 输出投影
        f = self.proj(h)                        # [B,T,feat_dim]
        f = self.norm(f)
        return f


class AetherBaseSpeechJSCC(nn.Module):
    """
    Aether-Base: 中杯老师模型

    架构与Lite相同，但容量更大，负责向Lite提供蒸馏目标
    """

    def __init__(self,
                 feat_dim=36,
                 d_csi=3,
                 d_z=16,
                 d_s=16,
                 n_bits=16,
                 hidden=128,
                 hash_method='bihalf',
                 device=None):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 中杯编解码器（容量比Lite大）
        self.enc = EncoderBase(feat_dim, d_csi, d_z, hidden)
        self.jscc_enc = JSCCEncoderBase(d_z, d_s, d_csi, hidden//2)
        self.jscc_dec = JSCCDecoderBase(d_z, d_s, d_csi, hidden//2)
        self.dec = DecoderBase(feat_dim, d_csi, d_z, hidden)

        # Hash瓶颈（复用final_version，容量稍大）
        self.hash = HashBottleneck(
            input_dim=d_z,
            hash_bits=n_bits,
            decoder_hidden=160,  # 比Lite的128更大
            output_dim=d_z,
            hash_method=hash_method,
            channel_type='bsc'
        )

        # Vocoder（复用final_version的FARGAN）
        self.vocoder = FARGANDecoder(
            fargan_subframe_size=40,
            fargan_nb_subframes=4,
            frame_rate_hz=100.0
        )

        # 配置
        self.feat_dim = feat_dim
        self.d_z = d_z
        self.n_bits = n_bits
        self.hidden = hidden

        # 移动到目标设备
        self.to(self.device)

    def forward_continuous(self, x_feat, csi, snr_db, channel_mode="awgn"):
        """连续JSCC模式"""
        # 编码
        z = self.enc(x_feat, csi)
        s = self.jscc_enc(z, csi)

        # 信道模拟
        if channel_mode == "awgn":
            y = channel_awgn(s, snr_db)
            h = None
        else:
            y, h = channel_rayleigh(s, snr_db)

        # 解码
        z_hat = self.jscc_dec(y, csi, h)
        feat_hat = self.dec(z_hat, csi)

        return feat_hat, z_hat, z

    def forward_hash(self, x_feat, csi, snr_db, channel_mode="awgn",
                    bit_noise=True, channel_params=None):
        """Hash + bit-level JSCC模式"""
        # 编码
        z = self.enc(x_feat, csi)
        s = self.jscc_enc(z, csi)

        # 信道模拟
        if channel_mode == "awgn":
            y = channel_awgn(s, snr_db)
            h = None
        else:
            y, h = channel_rayleigh(s, snr_db)

        z_hat = self.jscc_dec(y, csi, h)

        # Hash瓶颈
        if channel_params is None:
            channel_params = {'ber': 0.1}

        hash_output = self.hash(z_hat, channel_params if bit_noise else None)
        z_q = hash_output['reconstructed']

        # 额外的bit噪声
        if bit_noise and self.training:
            b_clean = hash_output['hash_bits_clean']
            b_noisy = apply_bit_noise(b_clean, flip_prob=0.05, block_drop_prob=0.02)
            z_q = self.hash.hash_decoder(b_noisy)

        # 解码
        feat_hat = self.dec(z_q, csi)

        return feat_hat, z_q, z_hat, hash_output

    def forward_full(self, x_feat, csi, snr_db, channel_mode="awgn",
                    target_len=None, with_hash=True):
        """完整前向传播"""
        if with_hash:
            feat_hat, z_q, z_hat, hash_output = self.forward_hash(
                x_feat, csi, snr_db, channel_mode, bit_noise=self.training
            )
        else:
            feat_hat, z_hat, z = self.forward_continuous(
                x_feat, csi, snr_db, channel_mode
            )
            hash_output = None

        # FARGAN声码器
        period, audio = self.vocoder(feat_hat, target_len=target_len)

        return {
            'audio': audio,
            'period': period,
            'feat_hat': feat_hat,
            'z_hat': z_hat if not with_hash else hash_output['reconstructed'],
            'hash_output': hash_output
        }

    def get_distillation_targets(self, x_feat, csi, snr_db, channel_mode="awgn"):
        """为Lite模型提供蒸馏目标"""
        with torch.no_grad():
            # 连续JSCC模式的输出
            feat_hat_base, z_hat_base, z_base = self.forward_continuous(
                x_feat, csi, snr_db, channel_mode
            )

            # 生成波形用于波形蒸馏
            period_base, audio_base = self.vocoder(feat_hat_base)

            return {
                'z_base': z_base,
                'z_hat_base': z_hat_base,
                'feat_hat_base': feat_hat_base,
                'audio_base': audio_base,
                'period_base': period_base
            }

    def get_bitrate(self, frame_rate: float = 100.0) -> float:
        """计算标称码率"""
        return self.n_bits * frame_rate / 1000.0  # kbps

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'AetherBaseSpeechJSCC',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024),
            'bitrate_kbps': self.get_bitrate(),
            'feature_dim': self.feat_dim,
            'latent_dim': self.d_z,
            'hash_bits': self.n_bits,
            'hidden_size': self.hidden,
            'device': str(self.device),
            'role': 'Medium Teacher (5-10M params)',
            'components': {
                'encoder': 'EncoderBase',
                'jscc_enc': 'JSCCEncoderBase',
                'jscc_dec': 'JSCCDecoderBase',
                'decoder': 'DecoderBase',
                'hash': 'HashBottleneck',
                'vocoder': 'FARGANDecoder'
            }
        }


def create_aether_base(feat_dim=36,
                      n_bits=16,
                      hidden=128,
                      device=None) -> AetherBaseSpeechJSCC:
    """便捷工厂函数"""
    return AetherBaseSpeechJSCC(
        feat_dim=feat_dim,
        d_csi=3,
        d_z=16,
        d_s=16,
        n_bits=n_bits,
        hidden=hidden,
        hash_method='bihalf',
        device=device
    )


if __name__ == "__main__":
    pass