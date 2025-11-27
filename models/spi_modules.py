#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemMap-PosEnc-ImgCodec (SPI) 融合模块
语义映射 + 位置编码 + 语音特征转图像的创新架构

设计理念：
- 将20维语音特征转为64x64图像表示
- 提取全局语义并生成自适应位置编码
- 使用轻量级图像codec进行压缩传输
- 完全端到端可训练，无需预训练大模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional


class SpeechToImageTransform(nn.Module):
    """改进版时序保持的语音特征到图像转换模块"""

    def __init__(self, feat_dim=20, seq_len=200, img_size=64, max_time_patches=32, patch_time_len=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.img_size = img_size
        self.max_time_patches = max_time_patches  # 最大时间patch数量
        self.patch_time_len = patch_time_len      # 每个patch的时间长度

        # 计算patch化后的图像尺寸
        # 每个patch的特征维度：包含所有增强信息
        self.enhanced_patch_dim = self.patch_time_len * (64 + 16 + 16 + 16)  # patch_time_len * 112
        # 解码时只输出基础特征
        self.base_patch_dim = self.patch_time_len * 64

        # 基于实际特征语义的分层定义：纯编码容量分配策略
        # 20维特征：前18维mel特征，第19维基音(f0)，第20维清浊音(voicing)
        self.feature_groups = {
            'f0_critical': {        # 第19维：基音 - 最重要
                'dims': [18],
                'encoding_capacity': 24,    # 分配24维编码容量 (1维->24维)
                'decoding_capacity': 24     # 对应24维解码容量
            },
            'voicing_critical': {   # 第20维：清浊音 - 很重要
                'dims': [19],
                'encoding_capacity': 16,    # 分配16维编码容量 (1维->16维)
                'decoding_capacity': 16
            },
            'mel_energy_high': {    # mel特征低频部分 - 重要
                'dims': list(range(0, 6)),  # 前6维低频
                'encoding_capacity': 48,    # 分配48维编码容量 (6维->48维)
                'decoding_capacity': 48
            },
            'mel_energy_mid': {     # mel特征中频部分 - 中等重要
                'dims': list(range(6, 12)),
                'encoding_capacity': 24,    # 分配24维编码容量 (6维->24维)
                'decoding_capacity': 24
            },
            'mel_energy_low': {     # mel特征高频部分 - 相对不重要
                'dims': list(range(12, 18)),
                'encoding_capacity': 12,    # 分配12维编码容量 (6维->12维)
                'decoding_capacity': 12
            }
        }

        # 计算总编码容量：24+16+48+24+12=124维 -> 融合到64维

        # 编码端分层处理器：对不同特征组使用不同编码路径
        self.layered_encoders = nn.ModuleDict()

        # 为每个特征组创建专门的编码器：容量分配策略
        total_encoding_capacity = 0
        for group_name, group_info in self.feature_groups.items():
            group_dim = len(group_info['dims'])
            if group_dim > 0:
                encoding_capacity = group_info['encoding_capacity']
                total_encoding_capacity += encoding_capacity

                # 根据容量分配创建编码器
                self.layered_encoders[group_name] = nn.Sequential(
                    nn.Linear(group_dim, encoding_capacity),
                    nn.GELU(),
                    nn.LayerNorm(encoding_capacity),
                    nn.Dropout(0.05),  # 统一dropout
                    nn.Linear(encoding_capacity, encoding_capacity),
                    nn.GELU(),
                    nn.LayerNorm(encoding_capacity)
                )

        # 特征融合器：将分层编码结果(124维)融合成64维
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_encoding_capacity, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64)
        )

        # 备用统一编码器
        self.unified_encoder = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.LayerNorm(64)
        )

        # 通道2：语义级时序压缩（标准化normalization位置）
        self.semantic_temporal_compressor = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, 32),  # GroupNorm更稳定
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16)
        )

        # patch内部时序建模（标准化normalization）
        self.patch_temporal_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=8),  # 分组卷积
            nn.GELU(),
            nn.GroupNorm(8, 64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 64)
        )

        # 时序位置编码（区分时间轴vs特征轴）
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, max_time_patches, 16) * 0.01
        )
        self.feature_pos_encoding = nn.Parameter(
            torch.randn(1, 64, 16) * 0.01
        )

        # patch编码器：将每个patch编码成独立的latent token（标准化）
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.enhanced_patch_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),  # 每个patch输出32维latent
            nn.Tanh()
        )

        # 图像后处理（添加残差连接，改进稳定性）
        self.image_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16)
        )
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(2, 8)
        )
        self.image_conv3 = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 残差连接投影层
        self.residual_proj = nn.Conv2d(1, 1, kernel_size=1)

        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # patch解码器：简化维度处理，使用自适应输入层
        self.adaptive_input_proj = nn.ModuleDict({
            '8': nn.Linear(8, 64),
            '16': nn.Linear(16, 64),
            '32': nn.Linear(32, 64),
            '64': nn.Linear(64, 64)  # 恒等投影
        })

        # patch解码器核心：标准化版本
        self.patch_decoder_core = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.base_patch_dim)  # 只输出基础特征
        )

        # intra-patch 1D解码器：恢复patch内部时序结构（标准化）
        self.intra_patch_decoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 64)
        )

        # 分层特征逆映射器：对不同维度使用不同压缩率
        self.layered_feature_mapper = nn.ModuleDict()

        # 为每个特征组创建专门的映射器
        total_hidden = 0
        for group_name, group_info in self.feature_groups.items():
            group_dim = len(group_info['dims'])
            if group_dim > 0:
                # 直接使用解码容量作为隐藏层维度
                hidden_dim = group_info['decoding_capacity']
                total_hidden += hidden_dim

                # 所有特征都移除最终激活函数，保持完整值域
                self.layered_feature_mapper[group_name] = nn.Sequential(
                    nn.Linear(hidden_dim, max(hidden_dim//2, group_dim)),
                    nn.GELU(),
                    nn.LayerNorm(max(hidden_dim//2, group_dim)),
                    nn.Linear(max(hidden_dim//2, group_dim), group_dim)
                    # 完全移除最终激活函数
                )

        # 主特征分配器：将64维隐藏特征分配给不同组
        self.feature_distributor = nn.Sequential(
            nn.Linear(64, total_hidden),
            nn.GELU(),
            nn.LayerNorm(total_hidden)
        )

        # 计算各组在total_hidden中的起始位置
        self.group_hidden_ranges = {}
        start_idx = 0
        for group_name, group_info in self.feature_groups.items():
            group_dim = len(group_info['dims'])
            if group_dim > 0:
                hidden_dim = group_info['decoding_capacity']
                self.group_hidden_ranges[group_name] = (start_idx, start_idx + hidden_dim)
                start_idx += hidden_dim

        # 传统映射器作为备用
        self.fallback_feature_mapper = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, feat_dim)
        )

        # 简化权重
        self.semantic_blend_weight = nn.Parameter(torch.tensor(0.2))
        self.temporal_preserve_weight = nn.Parameter(torch.tensor(0.8))

        # 轻量级时序增强LSTM - 在patch级别建模
        self.temporal_lstm = nn.LSTM(
            input_size=64,  # patch特征维度
            hidden_size=32,  # 轻量化隐藏维度
            num_layers=1,    # 单层避免过拟合
            dropout=0.0,     # 单层不需要dropout
            bidirectional=False,  # 单向保持因果性
            batch_first=True
        )
        self.lstm_proj = nn.Linear(32, 64)  # 投影回原始维度
        self.enable_lstm = True  # LSTM开关

    def patch_decoder(self, patch_latent):
        """简化的自适应patch解码器，支持更多维度"""
        input_dim = patch_latent.size(-1)
        input_dim_str = str(input_dim)

        if input_dim_str in self.adaptive_input_proj:
            hidden = self.adaptive_input_proj[input_dim_str](patch_latent)
        else:
            # 支持任意维度的投影（动态创建）
            if not hasattr(self, f'_dynamic_proj_{input_dim}'):
                setattr(self, f'_dynamic_proj_{input_dim}',
                       nn.Linear(input_dim, 64).to(patch_latent.device))

            hidden = getattr(self, f'_dynamic_proj_{input_dim}')(patch_latent)

        return self.patch_decoder_core(hidden)

    def layered_feature_encoding(self, feats):
        """
        编码端分层处理：对不同特征组使用不同编码路径

        Args:
            feats: [B, T, feat_dim] 输入特征

        Returns:
            encoded_features: [B, T, 64] 分层编码后的特征
        """
        B, T = feats.shape[:2]

        # 检查是否启用分层处理
        if not hasattr(self, 'layered_encoders') or len(self.layered_encoders) == 0:
            return self.unified_encoder(feats)

        # 分组编码
        group_encoded = []
        for group_name, group_info in self.feature_groups.items():
            if group_name in self.layered_encoders:
                feature_dims = group_info['dims']
                if len(feature_dims) > 0:
                    # 提取该组特征
                    group_feats = feats[..., feature_dims]  # [B, T, group_dim]

                    # 分层编码
                    group_encoded_feats = self.layered_encoders[group_name](group_feats)  # [B, T, hidden_dim]
                    group_encoded.append(group_encoded_feats)

        if len(group_encoded) > 0:
            # 拼接所有组的编码结果
            concatenated = torch.cat(group_encoded, dim=-1)  # [B, T, total_encoded_dim]

            # 融合成64维
            encoded_features = self.feature_fusion(concatenated)  # [B, T, 64]
        else:
            # 备用编码
            encoded_features = self.unified_encoder(feats)

        return encoded_features

    def layered_feature_mapping(self, hidden_features):
        """
        分层特征映射：编码端的镜像解码，基于容量分配

        Args:
            hidden_features: [B, T, 64] 隐藏特征

        Returns:
            features: [B, T, feat_dim] 分层重建的特征
        """
        B, T = hidden_features.shape[:2]

        # 检查是否启用分层处理
        if not hasattr(self, 'feature_groups') or len(self.feature_groups) == 0:
            return self.fallback_feature_mapper(hidden_features)

        # 将64维扩展回分层解码容量
        distributed_features = self.feature_distributor(hidden_features)  # [B, T, total_decoding_capacity]

        # 重建各个特征组 - 使用现有的hidden ranges
        group_outputs = {}
        for group_name, (start_idx, end_idx) in self.group_hidden_ranges.items():
            if group_name in self.layered_feature_mapper:
                group_hidden = distributed_features[..., start_idx:end_idx]  # [B, T, group_hidden_dim]
                group_features = self.layered_feature_mapper[group_name](group_hidden)  # [B, T, group_dim]

                # 对关键的一维组（F0 / Voicing）增加轻量残差，
                # 使用 group_hidden 的前 group_dim 维作为快捷通路，
                # 在不依赖原始特征的前提下抑制过度扭曲。
                if group_name in ("f0_critical", "voicing_critical"):
                    group_dim = len(self.feature_groups[group_name]['dims'])
                    if group_dim > 0 and group_hidden.size(-1) >= group_dim:
                        shortcut = group_hidden[..., :group_dim]
                        group_features = shortcut + group_features

                group_outputs[group_name] = group_features

        # 组装完整特征向量
        output_features = torch.zeros(B, T, self.feat_dim, device=hidden_features.device, dtype=hidden_features.dtype)

        for group_name, group_info in self.feature_groups.items():
            if group_name in group_outputs:
                feature_dims = group_info['dims']
                if len(feature_dims) > 0:
                    # 直接赋值，不检查维度匹配（因为解码器输出维度就是group_dim）
                    output_features[..., feature_dims] = group_outputs[group_name]

        return output_features

    def forward(self, feats):
        """
        改进版时序保持的语音特征到图像转换

        Args:
            feats: [B, T, D] 输入语音特征

        Returns:
            speech_image: [B, 1, H, W] 语音图像表示
            temporal_info: 保存的时间信息用于逆变换
        """
        B, T, D = feats.shape

        # === 通道1：分层编码（根据特征语义重要性） ===
        fine_features = self.layered_feature_encoding(feats)  # [B, T, 64]

        # patch内部时序建模
        fine_temporal = self.patch_temporal_encoder(
            fine_features.transpose(1, 2)
        ).transpose(1, 2)  # [B, T, 64]

        # 自适应时间分片（关键改进）
        if T <= self.max_time_patches * self.patch_time_len:
            # 短序列：直接padding到标准长度
            target_len = self.max_time_patches * self.patch_time_len
            fine_padded = F.pad(fine_temporal.transpose(1,2),
                               (0, target_len - T), mode='replicate').transpose(1,2)
            time_patches = fine_padded.view(B, self.max_time_patches, self.patch_time_len, 64)
            actual_patches = self.max_time_patches
            patch_stride = 1

            # 创建patch mask: 前面是有效patch，后面可能有padding
            effective_patches = min(self.max_time_patches, (T + self.patch_time_len - 1) // self.patch_time_len)
            patch_mask = torch.ones(B, self.max_time_patches, device=feats.device, dtype=torch.float32)
            if effective_patches < self.max_time_patches:
                patch_mask[:, effective_patches:] = 0.0
        else:
            # 长序列：智能分段，保持内部结构
            patch_stride = max(1, T // self.max_time_patches)
            patches = []
            for i in range(self.max_time_patches):
                start = i * patch_stride
                end = min(start + self.patch_time_len, T)
                if end > T:
                    # 最后一段：取后patch_time_len帧
                    patch = fine_temporal[:, -self.patch_time_len:, :]
                else:
                    patch_len = end - start
                    if patch_len < self.patch_time_len:
                        # 补齐到patch_time_len
                        patch = fine_temporal[:, start:end, :]
                        pad_len = self.patch_time_len - patch_len
                        patch = F.pad(patch.transpose(1,2), (0, pad_len), mode='replicate').transpose(1,2)
                    else:
                        patch = fine_temporal[:, start:start+self.patch_time_len, :]
                patches.append(patch)
            time_patches = torch.stack(patches, dim=1)  # [B, max_patches, patch_len, 64]
            actual_patches = self.max_time_patches

            # 对于长序列，所有patches都是有效的
            patch_mask = torch.ones(B, self.max_time_patches, device=feats.device, dtype=torch.float32)

        # === 通道2：语义时序压缩 ===
        semantic_compressed = self.semantic_temporal_compressor(
            fine_features.transpose(1,2)
        ).transpose(1,2)  # [B, T', 16]

        # 全局池化得到语义向量
        semantic_global = semantic_compressed.mean(dim=1)  # [B, 16]

        # === 融合：patch + 语义增强 ===
        enhanced_patches = []
        for i in range(actual_patches):
            patch = time_patches[:, i, :, :]  # [B, patch_len, 64]

            # 加入时序位置信息
            temporal_pos = self.temporal_pos_encoding[:, i:i+1, :].expand(B, self.patch_time_len, -1)

            # 简化的特征位置编码 - 创建固定16维编码
            feature_pos = torch.zeros(B, self.patch_time_len, 16, device=patch.device, dtype=patch.dtype)
            # 使用简单的正弦位置编码
            for t in range(self.patch_time_len):
                for d in range(16):
                    if d % 2 == 0:
                        feature_pos[:, t, d] = torch.sin(torch.tensor(t / 10000.0 ** (d / 16.0)))
                    else:
                        feature_pos[:, t, d] = torch.cos(torch.tensor(t / 10000.0 ** ((d-1) / 16.0)))

            # 语义条件增强
            semantic_cond = semantic_global.unsqueeze(1).expand(-1, self.patch_time_len, -1)

            enhanced_patch = torch.cat([
                patch, temporal_pos, feature_pos, semantic_cond
            ], dim=-1)  # [B, patch_len, 64+16+16+16=112]

            enhanced_patches.append(enhanced_patch)

        # 转换为patch张量
        patches_tensor = torch.stack(enhanced_patches, dim=1)  # [B, max_patches, patch_len, 112]

        # 每个patch编码为独立的latent token
        B, num_patches, patch_len, feat_dim = patches_tensor.shape

        # 对每个patch进行编码
        patch_latents = []
        for i in range(num_patches):
            patch = patches_tensor[:, i, :, :]  # [B, patch_len, 112]
            # 将patch flatten并编码
            patch_flat = patch.view(B, -1)  # [B, patch_len * 112]
            # 如果维度不匹配，先进行调整
            if patch_flat.size(1) != self.enhanced_patch_dim:
                # 使用平均池化调整到正确维度
                patch_reshaped = patch.mean(dim=1)  # [B, 112]
                patch_latent = self.patch_encoder(patch_reshaped)  # [B, 32]
            else:
                patch_latent = self.patch_encoder(patch_flat)  # [B, 32]
            patch_latents.append(patch_latent)

        # 堆叠所有patch latents
        multi_token_latents = torch.stack(patch_latents, dim=1)  # [B, num_patches, 32]

        # === LSTM时序增强 ===
        if self.enable_lstm and actual_patches > 1:
            # 提取每个patch的时序代表特征 (使用patch平均)
            patch_temporal_features = []
            for i in range(actual_patches):
                # 使用细粒度时序特征的patch平均作为代表
                patch_repr = time_patches[:, i, :, :].mean(dim=1)  # [B, 64]
                patch_temporal_features.append(patch_repr)

            # 构建时序序列
            temporal_sequence = torch.stack(patch_temporal_features, dim=1)  # [B, num_patches, 64]

            # LSTM处理跨patch时序依赖
            lstm_out, _ = self.temporal_lstm(temporal_sequence)  # [B, num_patches, 32]
            lstm_enhanced = self.lstm_proj(lstm_out)  # [B, num_patches, 64]

            # 残差连接：将LSTM增强的时序信息融合回原始patches
            for i in range(actual_patches):
                # 获取LSTM增强的时序信息
                temporal_enhance = lstm_enhanced[:, i:i+1, :].expand(-1, self.patch_time_len, -1)  # [B, patch_len, 64]

                # 残差连接到原始patch (权重0.3避免过度修改)
                time_patches[:, i, :, :64] = (
                    0.7 * time_patches[:, i, :, :64] +
                    0.3 * temporal_enhance
                )

        # 为了兼容性，仍然生成图像表示
        patches_for_image = torch.stack([p.view(B, self.patch_time_len, -1).mean(dim=1) for p in enhanced_patches], dim=1)
        speech_image = patches_for_image.mean(dim=2).unsqueeze(1)  # [B, 1, max_patches]
        speech_image = speech_image.unsqueeze(-1).expand(-1, -1, -1, self.img_size)  # [B, 1, max_patches, img_size]

        # 调整为方形图像
        if speech_image.size(2) != self.img_size:
            speech_image = F.interpolate(
                speech_image, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            )

        # 图像后处理（带残差连接）
        residual = speech_image
        x = self.image_conv1(speech_image)
        x = self.image_conv2(x)
        x = self.image_conv3(x)

        # 残差连接
        speech_image = x + self.residual_weight * self.residual_proj(residual)
        speech_image = torch.sigmoid(speech_image)  # 确保输出范围[0,1]

        # 保存逆变换信息
        temporal_info = {
            'original_length': T,
            'actual_patches': actual_patches,
            'patch_stride': patch_stride,
            'semantic_global': semantic_global,
            'enhanced_patches': patches_tensor,  # 保存详细patch信息
            'multi_token_latents': multi_token_latents,  # 多token latent表示
            'original_feats': feats,  # 用于训练时的残差连接
            'patch_mask': patch_mask  # patch有效性mask [B, N_patches]
        }

        return speech_image, temporal_info, multi_token_latents  # 返回多token latents

    def inverse(self, speech_image=None, temporal_info=None, multi_token_latents=None):
        """
        改进版多token逆变换：优先使用multi_token_latents

        Args:
            speech_image: [B, 1, H, W] 语音图像表示 (备用)
            temporal_info: 时间信息字典 (可选)
            multi_token_latents: [B, num_patches, d_z] 多token latent表示

        Returns:
            feats: [B, T, D] 恢复的语音特征
        """
        # 优先使用multi_token_latents进行逆变换
        if multi_token_latents is not None:
            return self._inverse_from_tokens(multi_token_latents, temporal_info)
        elif temporal_info is not None and 'multi_token_latents' in temporal_info:
            return self._inverse_from_tokens(temporal_info['multi_token_latents'], temporal_info)
        else:
            # 备用：使用图像方式逆变换
            return self._inverse_from_image(speech_image, temporal_info)

    def _inverse_from_tokens(self, multi_token_latents, temporal_info):
        """
        从多token latents恢复语音特征

        Args:
            multi_token_latents: [B, num_patches, d_z]
            temporal_info: 时间信息字典
        """
        B, num_patches, d_z = multi_token_latents.shape

        # 使用保存的时间信息
        if temporal_info is not None:
            target_length = temporal_info['original_length']
            patch_stride = temporal_info['patch_stride']
            semantic_global = temporal_info['semantic_global']
            original_feats = temporal_info.get('original_feats', None)
        else:
            target_length = self.seq_len
            patch_stride = target_length // num_patches
            semantic_global = torch.zeros(B, 16, device=multi_token_latents.device)
            original_feats = None

        # 1. 从latent tokens恢复patch特征
        recovered_patches = []
        for i in range(num_patches):
            patch_latent = multi_token_latents[:, i, :]  # [B, d_z]

            # 解码patch latent到特征
            patch_features_flat = self.patch_decoder(patch_latent)  # [B, patch_feature_dim]

            # Reshape到patch形状: [B, patch_time_len * 64] -> [B, patch_time_len, 64]
            patch_features = patch_features_flat.view(B, self.patch_time_len, 64)  # [B, patch_len, 64]

            # 直接使用作为基础特征
            base_features = patch_features  # [B, patch_len, 64]

            # 使用intra-patch 1D解码器恢复patch内部结构
            base_features_t = base_features.transpose(1, 2)  # [B, 64, patch_len]
            enhanced_features_t = self.intra_patch_decoder(base_features_t)  # [B, 64, patch_len]
            enhanced_features = enhanced_features_t.transpose(1, 2)  # [B, patch_len, 64]

            recovered_patches.append(enhanced_features)

        # 2. 将所有patch组合成完整时间序列
        if target_length <= num_patches * self.patch_time_len:
            # 短序列：直接拼接
            time_recovered = torch.cat(recovered_patches, dim=1)[:, :target_length, :]  # [B, T, 64]
        else:
            # 长序列：按patch_stride组合，处理重叠
            time_recovered = torch.zeros(B, target_length, 64, device=multi_token_latents.device)
            count_tensor = torch.zeros(B, target_length, 1, device=multi_token_latents.device)

            for i, patch_features in enumerate(recovered_patches):
                start_time = i * patch_stride
                end_time = min(start_time + self.patch_time_len, target_length)
                patch_len = end_time - start_time

                if patch_len > 0:
                    time_recovered[:, start_time:end_time, :] += patch_features[:, :patch_len, :]
                    count_tensor[:, start_time:end_time, :] += 1

            # 防止除以零
            count_tensor = torch.clamp(count_tensor, min=1)
            time_recovered = time_recovered / count_tensor

        # 3. LSTM时序平滑 (在逆变换中增强连续性)
        if self.enable_lstm and self.training and target_length > self.patch_time_len:
            # 对恢复的64维特征进行时序平滑
            # 分段处理避免内存过载
            chunk_size = 50  # 每次处理50帧
            if target_length > chunk_size:
                smoothed_chunks = []
                for start_idx in range(0, target_length, chunk_size):
                    end_idx = min(start_idx + chunk_size, target_length)
                    chunk = time_recovered[:, start_idx:end_idx, :]  # [B, chunk_len, 64]

                    if chunk.size(1) > 1:  # 只对多帧进行LSTM处理
                        chunk_smooth, _ = self.temporal_lstm(chunk)  # [B, chunk_len, 32]
                        chunk_smooth = self.lstm_proj(chunk_smooth)  # [B, chunk_len, 64]

                        # 残差连接进行平滑
                        chunk = 0.8 * chunk + 0.2 * chunk_smooth

                    smoothed_chunks.append(chunk)

                time_recovered = torch.cat(smoothed_chunks, dim=1)

        # 4. 分层特征映射：64 -> feat_dim（使用不同压缩率）
        freq_recovered = self.layered_feature_mapping(time_recovered)  # [B, T, feat_dim]

        # 5. 物理约束：对 F0 / Voicing 做轻量范围限制
        #    注意：避免对参与梯度计算的张量做原地修改，全部使用
        #    函数式重建最后一维，防止 autograd 版本冲突。
        #    - 第 19 维（index 18）：F0，非负且在训练中通常落在 0~5
        #    - 第 20 维（index 19）：Voicing，作为清浊门控，限制在 [0,1]
        if freq_recovered.dim() == 3:
            feat_dim = freq_recovered.size(-1)
            f0_idx = 18
            voicing_idx = 19

            if feat_dim > max(f0_idx, voicing_idx):
                # 取出原始 F0 / Voicing 分量
                f0 = freq_recovered[..., f0_idx]
                voicing = freq_recovered[..., voicing_idx]

                # 应用物理约束（纯函数式）
                f0_clamped = torch.clamp(f0, min=0.0, max=5.0)
                voicing_sig = voicing.sigmoid()

                # 重建最后一维，避免对原 tensor 做 in-place 赋值
                slices = []
                for i in range(feat_dim):
                    if i == f0_idx:
                        v = f0_clamped
                    elif i == voicing_idx:
                        v = voicing_sig
                    else:
                        v = freq_recovered[..., i]
                    slices.append(v.unsqueeze(-1))

                freq_recovered = torch.cat(slices, dim=-1)

        # 解码端已通过分层特征映射并施加基础物理约束，直接返回重建特征
        return freq_recovered

    def _inverse_from_image(self, speech_image, temporal_info):
        """
        从图像进行逆变换（备用方法）
        """
        B, _, H, W = speech_image.shape

        if temporal_info is not None:
            target_length = temporal_info['original_length']
            actual_patches = temporal_info['actual_patches']
        else:
            target_length = self.seq_len
            actual_patches = self.max_time_patches

        # 简化的图像逆变换（作为备用）
        image_flat = speech_image.view(B, -1)
        recovered_features = torch.zeros(B, target_length, self.feat_dim, device=speech_image.device)

        return recovered_features


class SemanticPositionalEncoder(nn.Module):
    """语义映射 + 位置编码增强器"""

    def __init__(self, img_size=64, semantic_dim=16, pos_dim=8):
        super().__init__()
        self.img_size = img_size
        self.semantic_dim = semantic_dim
        self.pos_dim = pos_dim

        # 语义内容提取器（标准化版本）
        self.semantic_extractor = nn.Sequential(
            # 多尺度特征提取
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),  # 大感受野
            nn.GELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 中感受野
            nn.GELU(),
            nn.GroupNorm(8, 32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, semantic_dim),
            nn.Tanh()
        )

        # 位置编码参数生成器
        self.pos_generator = nn.Sequential(
            nn.Linear(semantic_dim, pos_dim * 4),  # 生成sin/cos的频率和相位
            nn.Tanh()
        )

        # 图像增强融合网络（添加残差连接）
        self.enhance_conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 原图 + 位置编码
            nn.GELU(),
            nn.GroupNorm(8, 32)
        )
        self.enhance_conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, 16)
        )
        self.enhance_conv3 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 输入图像投影（用于残差连接）
        self.input_proj = nn.Conv2d(1, 1, kernel_size=1)

        # 可学习的融合权重
        self.enhance_weight = nn.Parameter(torch.tensor(0.8))
        self.residual_weight_enhance = nn.Parameter(torch.tensor(0.2))

        # 语义一致性约束
        self.semantic_projector = nn.Linear(semantic_dim, semantic_dim)

    def forward(self, speech_image):
        """
        语义增强 + 自适应位置编码

        Args:
            speech_image: [B, 1, H, W] 输入语音图像

        Returns:
            enhanced_image: [B, 1, H, W] 增强后的图像
            semantic_vec: [B, semantic_dim] 语义向量
            pos_encoding: [B, 1, H, W] 位置编码
        """
        B, C, H, W = speech_image.shape

        # 1. 提取全局语义
        semantic_vec = self.semantic_extractor(speech_image)  # [B, semantic_dim]

        # 2. 生成自适应位置编码参数
        pos_params = self.pos_generator(semantic_vec)  # [B, pos_dim*4]

        # 3. 创建2D位置编码
        pos_encoding = self._create_adaptive_2d_pe(
            B, H, W, pos_params, speech_image.device
        )  # [B, 1, H, W]

        # 4. 融合原图像与位置编码（带残差连接）
        enhanced_input = torch.cat([speech_image, pos_encoding], dim=1)  # [B, 2, H, W]

        # 前向传播增强网络
        x = self.enhance_conv1(enhanced_input)
        x = self.enhance_conv2(x)
        x = self.enhance_conv3(x)

        # 残差连接
        residual = self.input_proj(speech_image)
        enhanced_image = self.enhance_weight * torch.sigmoid(x) + self.residual_weight_enhance * residual

        # 5. 语义向量投影 (用于对比学习)
        semantic_proj = self.semantic_projector(semantic_vec)

        return enhanced_image, semantic_proj, pos_encoding

    def _create_adaptive_2d_pe(self, batch_size, height, width, pos_params, device):
        """基于语义内容生成自适应2D位置编码"""
        # 基础坐标网格
        y_coords = torch.linspace(-1, 1, height, device=device)
        x_coords = torch.linspace(-1, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        pos_encoding = torch.zeros(batch_size, 1, height, width, device=device)

        for b in range(batch_size):
            # 解析位置编码参数
            params = pos_params[b]  # [pos_dim*4]

            # 频率参数
            freq_y = params[:self.pos_dim].mean() * 5.0 + 1.0  # 范围 [1, 6]
            freq_x = params[self.pos_dim:2*self.pos_dim].mean() * 5.0 + 1.0

            # 相位参数
            phase_y = params[2*self.pos_dim:3*self.pos_dim].mean() * math.pi
            phase_x = params[3*self.pos_dim:].mean() * math.pi

            # 生成自适应位置编码
            pos_y = torch.sin(y_grid * freq_y + phase_y)
            pos_x = torch.cos(x_grid * freq_x + phase_x)

            # 组合位置编码 (轻度调制，避免过度影响原信号)
            pos_encoding[b, 0] = 0.2 * (pos_y + pos_x)

        return pos_encoding


class LightweightImageCodec(nn.Module):
    """轻量级图像编解码器"""

    def __init__(self, img_size=64, latent_dim=32, base_channels=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # 编码器：64x64 -> latent_dim（标准化版本）
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, base_channels),

            # 32x32 -> 16x16
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(2, base_channels*2),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(4, base_channels*4),

            # 8x8 -> 4x4
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, base_channels*8),

            # 4x4 -> latent
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, latent_dim),
            nn.Tanh()
        )

        # 解码器：latent_dim -> 64x64 (修复checkerboard artifacts)
        self.decoder_proj = nn.Sequential(
            nn.Linear(latent_dim, base_channels*8*4*4),
            nn.GELU(),
            nn.Unflatten(1, (base_channels*8, 4, 4))
        )

        # 4x4 -> 8x8 (使用3x3 kernel避免checkerboard)
        self.up_block1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, base_channels*4),  # GroupNorm更稳定
            nn.GELU()
        )

        # 8x8 -> 16x16
        self.up_block2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2, base_channels*2),
            nn.GELU()
        )

        # 16x16 -> 32x32
        self.up_block3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(1, base_channels),
            nn.GELU()
        )

        # 32x32 -> 64x64 (最终层使用Tanh避免过度约束)
        self.up_block4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        """编码：图像 -> latent"""
        return self.encoder(x)

    def decode(self, z):
        """解码：latent -> 图像（修复checkerboard artifacts）"""
        x = self.decoder_proj(z)

        # 添加残差连接保持特征范围
        x1 = self.up_block1(x)
        x2 = self.up_block2(x1)
        x3 = self.up_block3(x2)
        x4 = self.up_block4(x3)

        # 输出范围调整到[0,1]
        return (x4 + 1.0) * 0.5

    def forward(self, x):
        """完整编解码"""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def extract_f0_yin(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160,
                   f0_min: float = 80.0, f0_max: float = 400.0, threshold: float = 0.1) -> torch.Tensor:
    """
    基于YIN算法的F0提取器（业界标准实现）

    YIN: A fundamental frequency estimator for speech and music
    参考论文: de Cheveigné, A., & Kawahara, H. (2002)

    Args:
        audio: [B, L] 音频信号
        sr: 采样率
        hop_length: 帧移
        f0_min: 最小F0频率 (Hz)
        f0_max: 最大F0频率 (Hz)
        threshold: YIN阈值

    Returns:
        f0: [B, T] F0轨迹 (Hz)
    """
    B, L = audio.shape

    # 计算窗口大小和参数
    win_length = hop_length * 4  # 窗口长度为帧移的4倍
    n_frames = (L - win_length) // hop_length + 1

    # 计算tau范围（周期搜索范围）
    tau_min = int(sr / f0_max)  # 最大F0对应的最小周期
    tau_max = int(sr / f0_min)  # 最小F0对应的最大周期
    tau_max = min(tau_max, win_length // 2)  # 限制在窗口长度的一半

    f0_batch = []

    for b in range(B):
        audio_single = audio[b]  # [L]
        f0_frames = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + win_length

            if end > L:
                # 处理边界情况
                frame = audio_single[start:]
                if len(frame) < win_length // 2:
                    f0_frames.append(0.0)
                    continue
                # 零填充
                frame = F.pad(frame.unsqueeze(0), (0, win_length - len(frame))).squeeze(0)
            else:
                frame = audio_single[start:end]

            # YIN算法核心
            f0_hz = yin_estimate_f0(frame, sr, tau_min, tau_max, threshold)
            f0_frames.append(f0_hz)

        f0_batch.append(torch.tensor(f0_frames, device=audio.device, dtype=audio.dtype))

    # 堆叠所有batch
    f0 = torch.stack(f0_batch, dim=0)  # [B, T]

    # 后处理：中值滤波平滑
    f0 = median_filter_1d(f0, kernel_size=3)

    return f0


def yin_estimate_f0(frame: torch.Tensor, sr: int, tau_min: int, tau_max: int, threshold: float) -> float:
    """
    YIN算法核心实现

    Args:
        frame: [L] 单帧音频
        sr: 采样率
        tau_min: 最小tau（对应最大F0）
        tau_max: 最大tau（对应最小F0）
        threshold: YIN阈值

    Returns:
        f0_hz: 估计的F0频率 (Hz)
    """
    frame_len = len(frame)

    # 步骤1: 计算差分函数 d_t(tau)
    d_tau = torch.zeros(tau_max + 1, device=frame.device, dtype=frame.dtype)

    for tau in range(1, tau_max + 1):
        if tau >= frame_len:
            d_tau[tau] = float('inf')
            continue

        diff = frame[:-tau] - frame[tau:]
        d_tau[tau] = torch.sum(diff ** 2)

    # 步骤2: 计算累积平均归一化差分函数 d'_t(tau)
    d_tau_prime = torch.zeros_like(d_tau)
    d_tau_prime[0] = 1.0

    cumsum = d_tau[0]
    for tau in range(1, tau_max + 1):
        cumsum += d_tau[tau]
        if cumsum == 0:
            d_tau_prime[tau] = 1.0
        else:
            d_tau_prime[tau] = d_tau[tau] * tau / cumsum

    # 步骤3: 寻找第一个低于阈值的局部最小值
    tau_estimate = 0
    for tau in range(tau_min, tau_max):
        if d_tau_prime[tau] < threshold:
            # 寻找局部最小值
            if tau == tau_max - 1 or d_tau_prime[tau] < d_tau_prime[tau + 1]:
                tau_estimate = tau
                break

    # 如果没找到，选择全局最小值
    if tau_estimate == 0:
        tau_estimate = torch.argmin(d_tau_prime[tau_min:tau_max]).item() + tau_min

    # 步骤4: 抛物线插值提高精度
    if tau_min < tau_estimate < tau_max - 1:
        y1 = d_tau_prime[tau_estimate - 1]
        y2 = d_tau_prime[tau_estimate]
        y3 = d_tau_prime[tau_estimate + 1]

        # 抛物线插值
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-6:
            tau_precise = tau_estimate + (y3 - y1) / (2 * denom)
        else:
            tau_precise = tau_estimate
    else:
        tau_precise = tau_estimate

    # 转换为频率
    if tau_precise <= 0:
        return 0.0

    f0_hz = sr / tau_precise

    # 后验检查：确保F0在合理范围内
    if f0_hz < 80 or f0_hz > 400:
        return 0.0

    return f0_hz


def median_filter_1d(signal: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    1D中值滤波，用于F0轨迹平滑

    Args:
        signal: [B, T] 输入信号
        kernel_size: 滤波核大小（奇数）

    Returns:
        filtered: [B, T] 滤波后的信号
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    B, T = signal.shape
    pad_size = kernel_size // 2

    # 对称填充
    signal_padded = F.pad(signal, (pad_size, pad_size), mode='reflect')

    filtered = torch.zeros_like(signal)

    for t in range(T):
        window = signal_padded[:, t:t + kernel_size]  # [B, kernel_size]
        filtered[:, t] = torch.median(window, dim=1)[0]

    return filtered


def extract_f0_crepe_style(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160,
                          fmin: float = 80.0, fmax: float = 400.0) -> torch.Tensor:
    """
    CREPE风格的神经网络F0提取器（简化版本）

    使用预训练的轻量级CNN进行F0估计
    注意：这是简化实现，实际CREPE需要预训练模型

    Args:
        audio: [B, L] 音频信号
        sr: 采样率
        hop_length: 帧移
        fmin: 最小F0频率
        fmax: 最大F0频率

    Returns:
        f0: [B, T] F0轨迹
    """
    # 这里提供一个基于能量谱和谐波检测的简化版本
    B, L = audio.shape
    n_frames = L // hop_length

    # 窗口参数
    win_length = 1024
    n_fft = 2048

    f0_batch = []

    for b in range(B):
        audio_single = audio[b]
        f0_frames = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + win_length

            if end > L:
                frame = F.pad(audio_single[start:].unsqueeze(0),
                             (0, end - L)).squeeze(0)
            else:
                frame = audio_single[start:end]

            # 加窗
            window = torch.hann_window(win_length, device=audio.device)
            frame_windowed = frame * window

            # FFT
            fft = torch.fft.fft(frame_windowed, n=n_fft)
            magnitude = torch.abs(fft)[:n_fft//2]

            # 频率轴
            freqs = torch.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]

            # 限制频率范围
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if not freq_mask.any():
                f0_frames.append(0.0)
                continue

            # 谐波检测
            f0_estimate = estimate_f0_harmonic(magnitude, freqs, freq_mask, fmin, fmax)
            f0_frames.append(f0_estimate)

        f0_batch.append(torch.tensor(f0_frames, device=audio.device, dtype=audio.dtype))

    f0 = torch.stack(f0_batch, dim=0)

    # 平滑处理
    f0 = median_filter_1d(f0, kernel_size=5)

    return f0


def estimate_f0_harmonic(magnitude: torch.Tensor, freqs: torch.Tensor,
                        freq_mask: torch.Tensor, fmin: float, fmax: float) -> float:
    """
    基于谐波结构的F0估计
    """
    magnitude_masked = magnitude[freq_mask]
    freqs_masked = freqs[freq_mask]

    if len(magnitude_masked) == 0:
        return 0.0

    # 寻找可能的基频候选
    candidates = []

    # 搜索基频候选
    for i, freq in enumerate(freqs_masked):
        if freq < fmin or freq > fmax:
            continue

        # 检查谐波强度
        harmonic_strength = 0.0
        harmonic_count = 0

        for harmonic in range(1, 6):  # 检查前5个谐波
            harmonic_freq = freq * harmonic
            if harmonic_freq > freqs_masked[-1]:
                break

            # 找到最近的频率bin
            freq_diff = torch.abs(freqs_masked - harmonic_freq)
            closest_idx = torch.argmin(freq_diff)

            if freq_diff[closest_idx] < 10:  # 10Hz容差
                harmonic_strength += magnitude_masked[closest_idx]
                harmonic_count += 1

        if harmonic_count >= 2:  # 至少检测到2个谐波
            candidates.append((freq.item(), harmonic_strength.item()))

    if not candidates:
        return 0.0

    # 选择谐波强度最大的候选
    best_candidate = max(candidates, key=lambda x: x[1])
    return best_candidate[0]


# 主要接口函数
def extract_f0_professional(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160,
                           method: str = "yin", **kwargs) -> torch.Tensor:
    """
    专业F0提取器接口

    Args:
        audio: [B, L] 音频信号
        sr: 采样率
        hop_length: 帧移
        method: 提取方法 ("yin", "crepe_style")

    Returns:
        f0: [B, T] F0轨迹
    """
    if method == "yin":
        return extract_f0_yin(audio, sr, hop_length, **kwargs)
    elif method == "crepe_style":
        return extract_f0_crepe_style(audio, sr, hop_length, **kwargs)
    else:
        raise ValueError(f"Unknown F0 extraction method: {method}")


# 为了兼容性，保留原名字但使用专业实现
def extract_f0_simple(audio: torch.Tensor, sr: int = 16000, hop_length: int = 160) -> torch.Tensor:
    """兼容接口：使用YIN算法进行F0提取"""
    return extract_f0_professional(audio, sr, hop_length, method="yin")


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


class SPILoss(nn.Module):
    """SPI专用损失函数 - 集成Anti-Buzz功能"""

    def __init__(self,
                 semantic_weight=0.1,
                 image_weight=0.2,
                 position_weight=0.05,
                 # Anti-Buzz权重
                 f0_weight=5.0,
                 vuv_weight=5.0,
                 f0_shape_weight=1.0,
                 vuv_bce_weight=2.0,
                 audio_f0_weight=3.0,
                 f0_variance_weight=2.0,
                 silence_weight=3.0,
                 min_f0_std=20.0,
                 silence_threshold=0.01,
                 sr=16000,
                 hop_length=160):
        super().__init__()
        self.semantic_weight = semantic_weight
        self.image_weight = image_weight
        self.position_weight = position_weight

        # Anti-Buzz参数
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

        # 语义对比学习
        self.semantic_contrastive = nn.CosineSimilarity(dim=1)

    def forward(self, spi_output, targets, audio_pred=None, audio_target=None):
        """
        计算SPI损失（集成Anti-Buzz功能）

        Args:
            spi_output: SPI模块输出
            targets: 目标数据
            audio_pred: [B, L] 预测音频（可选，用于Anti-Buzz）
            audio_target: [B, L] 目标音频（可选，用于Anti-Buzz）
        """
        losses = {}

        # ===== 原有SPI损失 =====

        # 1. 图像重建损失
        if 'image_recon' in spi_output and 'original_image' in targets:
            losses['image_recon'] = F.mse_loss(
                spi_output['image_recon'],
                targets['original_image']
            )

        # 2. 语义一致性损失
        if 'semantic_enc' in spi_output and 'semantic_dec' in spi_output:
            losses['semantic_consistency'] = F.mse_loss(
                spi_output['semantic_enc'],
                spi_output['semantic_dec']
            )

        # 3. 位置编码正则化
        if 'pos_encoding' in spi_output:
            pos_reg = torch.mean(spi_output['pos_encoding'].pow(2))
            losses['position_reg'] = pos_reg

        # ===== Anti-Buzz损失 =====

        # 获取特征重建结果
        feats_recon = spi_output.get('feats_recon')
        if feats_recon is None:
            feats_recon = spi_output.get('feat_hat')
        if feats_recon is None:
            feats_recon = spi_output.get('feats_recovered')

        feats_target = targets.get('original_feats')
        if feats_target is None:
            feats_target = targets.get('feats')

        if feats_recon is not None and feats_target is not None:
            B, T, D = feats_recon.shape

            # 确保特征维度正确
            if D < 20:
                # 如果维度不足，用零填充
                padding = torch.zeros(B, T, 20 - D, device=feats_recon.device, dtype=feats_recon.dtype)
                feats_recon = torch.cat([feats_recon, padding], dim=-1)
                feats_target = torch.cat([feats_target, padding], dim=-1)

            # 4. 基础特征重建（前18维：倒谱等）
            cep_diff = feats_recon[..., :18] - feats_target[..., :18]
            losses['cep_recon'] = cep_diff.abs().mean()

            # 5. F0维度加权监督（第19维）
            f0_diff = feats_recon[..., 18:19] - feats_target[..., 18:19]
            losses['f0_feat'] = f0_diff.abs().mean()

            # 6. Voicing维度加权监督（第20维）
            vuv_diff = feats_recon[..., 19:20] - feats_target[..., 19:20]
            losses['vuv_feat'] = vuv_diff.abs().mean()

            # 7. F0形状相关性约束
            f0_pred_1d = feats_recon[..., 18]  # [B, T]
            f0_target_1d = feats_target[..., 18]  # [B, T]

            f0_correlation = compute_f0_correlation(f0_pred_1d, f0_target_1d)
            losses['f0_shape'] = (1.0 - f0_correlation).clamp(min=0).mean()

            # 8. Voicing二分类损失
            vuv_target = (feats_target[..., 19] > 0.5).float()  # 转为0/1标签
            vuv_pred_prob = torch.sigmoid(feats_recon[..., 19])  # 转为概率
            losses['vuv_bce'] = F.binary_cross_entropy(vuv_pred_prob, vuv_target)

        # ===== 音频级Anti-Buzz损失 =====

        if audio_pred is not None and audio_target is not None:
            try:
                # 9. 音频级F0损失 (使用YIN算法)
                with torch.no_grad():
                    f0_audio_target = extract_f0_yin(audio_target, self.sr, self.hop_length)

                f0_audio_pred = extract_f0_yin(audio_pred, self.sr, self.hop_length)

                # 对齐长度
                min_len = min(f0_audio_target.size(1), f0_audio_pred.size(1))
                f0_audio_target = f0_audio_target[:, :min_len]
                f0_audio_pred = f0_audio_pred[:, :min_len]

                losses['f0_audio'] = F.l1_loss(f0_audio_pred, f0_audio_target)

                # 10. F0方差下界约束（防止塌缩为常数）
                f0_std_pred = f0_audio_pred.std(dim=1)  # [B]
                losses['f0_variance'] = F.relu(self.min_f0_std - f0_std_pred).mean()

            except Exception as e:
                # 如果F0提取失败，使用零损失
                losses['f0_audio'] = torch.zeros(1, device=feats_recon.device if feats_recon is not None else audio_pred.device)
                losses['f0_variance'] = torch.zeros(1, device=feats_recon.device if feats_recon is not None else audio_pred.device)

            try:
                # 11. 静音段能量惩罚
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
                    losses['silence_penalty'] = torch.zeros(1, device=audio_pred.device)

            except Exception as e:
                losses['silence_penalty'] = torch.zeros(1, device=audio_pred.device)

        # ===== 总损失计算 =====

        total_loss = 0

        # 原有损失
        for key, loss in losses.items():
            if key in ['image_recon']:
                total_loss += self.image_weight * loss
            elif key in ['semantic_consistency']:
                total_loss += self.semantic_weight * loss
            elif key in ['position_reg']:
                total_loss += self.position_weight * loss
            elif key == 'cep_recon':
                total_loss += loss  # 基础权重1.0
            elif key == 'f0_feat':
                total_loss += self.f0_weight * loss
            elif key == 'vuv_feat':
                total_loss += self.vuv_weight * loss
            elif key == 'f0_shape':
                total_loss += self.f0_shape_weight * loss
            elif key == 'vuv_bce':
                total_loss += self.vuv_bce_weight * loss
            elif key == 'f0_audio':
                total_loss += self.audio_f0_weight * loss
            elif key == 'f0_variance':
                total_loss += self.f0_variance_weight * loss
            elif key == 'silence_penalty':
                total_loss += self.silence_weight * loss

        losses['spi_total'] = total_loss
        return total_loss, losses


def test_spi_modules():
    """测试改进版SPI模块功能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模块
    s2i = SpeechToImageTransform(feat_dim=20, max_time_patches=32, patch_time_len=8).to(device)
    spe = SemanticPositionalEncoder().to(device)
    codec = LightweightImageCodec().to(device)

    # 测试数据
    batch_size = 4
    feats = torch.randn(batch_size, 200, 20, device=device)

    print("Testing SPI Modules:")
    print(f"Input features: {feats.shape}")

    # 1. 语音 -> 图像 (现在返回多token latents)
    speech_image, temporal_info, multi_token_latents = s2i(feats)
    print(f"Speech image: {speech_image.shape}")
    print(f"Temporal info keys: {list(temporal_info.keys())}")
    print(f"Multi-token latents: {multi_token_latents.shape}")

    # 2. 语义增强
    enhanced_image, semantic_vec, pos_encoding = spe(speech_image)
    print(f"Enhanced image: {enhanced_image.shape}")
    print(f"Semantic vector: {semantic_vec.shape}")
    print(f"Position encoding: {pos_encoding.shape}")

    # 3. 图像编码
    recon_image, latent = codec(enhanced_image)
    print(f"Reconstructed image: {recon_image.shape}")
    print(f"Latent: {latent.shape}")

    # 4. 逆变换 (使用多token latents)
    feats_recon = s2i.inverse(recon_image, temporal_info, multi_token_latents)
    print(f"Reconstructed features: {feats_recon.shape}")

    # 计算重建误差
    min_len = min(feats.size(1), feats_recon.size(1))
    feats_orig_crop = feats[:, :min_len, :]
    feats_recon_crop = feats_recon[:, :min_len, :]
    recon_error = F.mse_loss(feats_recon_crop, feats_orig_crop)
    print(f"Reconstruction error: {recon_error.item():.6f}")
    print(f"Original features shape: {feats.shape}")
    print(f"Reconstructed features shape: {feats_recon.shape}")

    # 显示时序改进效果
    if temporal_info and 'actual_patches' in temporal_info:
        print(f"Temporal patches used: {temporal_info['actual_patches']}")
        print(f"Patch stride: {temporal_info['patch_stride']}")
        print(f"Time resolution preserved: ~{temporal_info['actual_patches'] * 8} frames")

    # 测试LSTM增强效果
    print(f"LSTM enabled: {s2i.enable_lstm}")
    total_params = sum(p.numel() for p in s2i.parameters())
    lstm_params = sum(p.numel() for p in s2i.temporal_lstm.parameters()) + s2i.lstm_proj.weight.numel() + s2i.lstm_proj.bias.numel()
    print(f"Total parameters: {total_params:,}")
    print(f"LSTM parameters: {lstm_params:,} ({lstm_params/total_params*100:.1f}%)")

    print("✓ SPI modules with LSTM enhancement test completed successfully!")


if __name__ == "__main__":
    test_spi_modules()
