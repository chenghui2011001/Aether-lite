#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPI-LiteSpeechJSCC：集成语义映射+位置编码+图像编码的完整JSCC系统

核心创新：
- 将20维语音特征转为图像表示
- 语义感知的位置编码
- 轻量级图像压缩传输
- 端到端可训练，推理时无大模型依赖
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .lite_speech_jscc import LiteSpeechJSCC
from .spi_modules import (
    SpeechToImageTransform,
    SemanticPositionalEncoder,
    LightweightImageCodec,
    SPILoss
)


class SPI_LiteSpeechJSCC(LiteSpeechJSCC):
    """集成SPI的LiteSpeechJSCC"""

    def __init__(
        self,
        feat_dim=20,
        d_csi=4,
        d_z=32,  # 扩大latent空间
        d_s=24,  # 对应调整symbol空间
        n_bits=24,
        hidden=80,
        img_size=64,
        semantic_dim=16,
        device='cuda',
        **kwargs
    ):
        # 使用扩大的参数初始化基类
        super().__init__(
            feat_dim=feat_dim,
            d_csi=d_csi,
            d_z=d_z,
            d_s=d_s,
            n_bits=n_bits,
            hidden=hidden,
            device=device,
            **kwargs
        )

        self.img_size = img_size
        self.semantic_dim = semantic_dim

        # === SPI核心组件 ===
        self.speech_to_image = SpeechToImageTransform(
            feat_dim=feat_dim,
            seq_len=200,  # 假设固定序列长度
            img_size=img_size,
            max_time_patches=32,  # 时序patch数量
            patch_time_len=8      # 每个patch的时间长度
        )

        self.semantic_pos_encoder = SemanticPositionalEncoder(
            img_size=img_size,
            semantic_dim=semantic_dim
        )

        self.image_codec = LightweightImageCodec(
            img_size=img_size,
            latent_dim=d_z  # 图像latent直接作为JSCC输入
        )

        # === 重新设计编解码器 ===
        # SPI编码器：不再直接处理特征，而是处理图像latent（标准化）
        self.spi_encoder = nn.Sequential(
            nn.Linear(d_z + d_csi, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_z),
            nn.Tanh()
        )

        # SPI解码器：从图像latent恢复特征（标准化）
        self.spi_decoder = nn.Sequential(
            nn.Linear(d_z + d_csi + semantic_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_z),
            nn.Tanh()
        )

        # 语义保持器（标准化）
        self.semantic_keeper = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.GELU(),
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim),
            nn.Tanh()
        )

        # SPI专用损失（集成Anti-Buzz功能）
        self.spi_loss = SPILoss(
            semantic_weight=0.1,
            image_weight=0.2,
            position_weight=0.05,
            # Anti-Buzz权重配置
            f0_weight=5.0,              # F0特征加权
            vuv_weight=5.0,             # Voicing特征加权
            f0_shape_weight=1.0,        # F0形状相关性
            vuv_bce_weight=2.0,         # Voicing二分类
            audio_f0_weight=3.0,        # 音频级F0损失
            f0_variance_weight=2.0,     # F0方差下界
            silence_weight=3.0,         # 静音段惩罚
            min_f0_std=20.0,           # 最小F0标准差（Hz）
            silence_threshold=0.01,     # 静音阈值
            sr=16000,
            hop_length=160
        )

        # 替换原有编解码器
        self.enc = self.spi_encoder
        self.dec = self.spi_decoder

    def spi_encode(self, feats: torch.Tensor, csi_vec: torch.Tensor) -> Dict:
        """
        SPI编码过程：特征 -> 图像 -> 语义增强 -> 压缩

        Args:
            feats: [B, T, feat_dim] 输入特征
            csi_vec: [B, d_csi] 信道状态信息

        Returns:
            字典包含所有中间结果
        """
        B, T, D = feats.shape

        # 1. 语音特征 -> 多个token latents (支持时间信息保存)
        speech_image, temporal_info, multi_token_latents = self.speech_to_image(feats)  # [B, 1, img_size, img_size], temporal_info, [B, num_patches, 32]

        # 2. 语义映射 + 位置编码增强
        enhanced_image, semantic_vec, pos_encoding = self.semantic_pos_encoder(speech_image)

        # 3. 图像压缩编码
        image_latent = self.image_codec.encode(enhanced_image)  # [B, d_z]

        # 4. 多个token的CSI融合编码
        B, num_patches, token_dim = multi_token_latents.shape
        csi_expanded = csi_vec.unsqueeze(1).expand(B, num_patches, -1)  # [B, num_patches, d_csi]

        # 每个token都与CSI融合
        tokens_with_csi = torch.cat([multi_token_latents, csi_expanded], dim=-1)  # [B, num_patches, token_dim + d_csi]

        # 逐个token编码 (简化处理：直接使用原始多token latents作为JSCC输入)
        z_encoded_multi = multi_token_latents  # [B, num_patches, token_dim]

        # 保持单一latent兼容性
        latent_csi = torch.cat([image_latent, csi_vec], dim=-1)  # [B, d_z + d_csi]
        z_encoded = self.spi_encoder(latent_csi)  # [B, d_z]

        return {
            'z_encoded': z_encoded,              # 单一latent(兼容性)
            'z_encoded_multi': z_encoded_multi,  # 多个token latents
            'multi_token_latents': multi_token_latents,  # 原始多个token
            'image_latent': image_latent,        # 原始图像latent
            'semantic_vec': semantic_vec,        # 语义向量
            'speech_image': speech_image,        # 原始语音图像
            'enhanced_image': enhanced_image,    # 增强语音图像
            'pos_encoding': pos_encoding,        # 位置编码
            'temporal_info': temporal_info,      # 时间信息 (用于逆变换)
        }

    def spi_decode(self, z_decoded_multi: torch.Tensor, csi_vec: torch.Tensor, semantic_vec: torch.Tensor, temporal_info: Dict = None, z_decoded_single=None) -> Dict:
        """
        SPI解码过程：多个token解压 -> 特征重建

        Args:
            z_decoded_multi: [B, num_patches, d_z] JSCC解码的多个token
            csi_vec: [B, d_csi] 信道状态信息
            semantic_vec: [B, semantic_dim] 语义向量
            z_decoded_single: [B, d_z] 单一latent(备用)

        Returns:
            字典包含恢复的特征和图像
        """
        # 1. 语义保持处理
        semantic_kept = self.semantic_keeper(semantic_vec)  # [B, semantic_dim]

        if z_decoded_multi is not None:
            # 检查输入tensor的维度
            if z_decoded_multi.dim() == 2:
                # 输入是[B, d_z]，转换为[B, 1, d_z]以保持一致性
                B, d_z = z_decoded_multi.shape
                z_decoded_multi = z_decoded_multi.unsqueeze(1)  # [B, 1, d_z]
                num_patches = 1
            elif z_decoded_multi.dim() == 3:
                # 输入已经是[B, num_patches, d_z]
                B, num_patches, d_z = z_decoded_multi.shape
            else:
                raise ValueError(f"Expected z_decoded_multi to be 2D or 3D tensor, got {z_decoded_multi.dim()}D")

            # 2. 直接从多个token恢复特征
            feats_recovered = self.speech_to_image.inverse(
                multi_token_latents=z_decoded_multi,
                temporal_info=temporal_info
            )

            # 3. 为了兼容性，仍然生成图像表示
            image_latent_recovered = z_decoded_multi[:, 0, :]  # 使用第一个token作为代表
            image_recovered = self.image_codec.decode(image_latent_recovered)
        else:
            # 备用：单一latent解码流程
            B = z_decoded_single.size(0)
            decoder_input = torch.cat([z_decoded_single, csi_vec, semantic_kept], dim=-1)
            image_latent_recovered = self.spi_decoder(decoder_input)
            image_recovered = self.image_codec.decode(image_latent_recovered)
            feats_recovered = self.speech_to_image.inverse(image_recovered, temporal_info)

        return {
            'feats_recovered': feats_recovered,
            'image_recovered': image_recovered,
            'image_latent_recovered': image_latent_recovered,
            'semantic_kept': semantic_kept,
        }

    def forward_spi_stage2(self, feats: torch.Tensor, csi_vec: torch.Tensor) -> Dict:
        """
        SPI版本的Stage2前向传播：连续JSCC

        Args:
            feats: [B, T, feat_dim] 输入特征
            csi_vec: [B, d_csi] 信道状态信息

        Returns:
            包含所有输出的字典
        """
        # === 编码阶段 ===
        encode_out = self.spi_encode(feats, csi_vec)
        z = encode_out['z_encoded']
        z_multi = encode_out['z_encoded_multi']  # 多token latents
        semantic_vec = encode_out['semantic_vec']
        temporal_info = encode_out['temporal_info']  # 提取时间信息

        # === 多token JSCC传输 ===
        # 对多个token进行JSCC编码
        B, num_patches, token_dim = z_multi.shape
        s_multi = self.jscc_enc(z_multi, csi_vec)  # JSCC编码 [B, num_patches, d_s]
        z_hat_multi = self.jscc_dec(s_multi, csi_vec)  # JSCC解码 [B, num_patches, d_z]

        # 保持单一latent兼容性
        z_seq = z.unsqueeze(1)  # [B, d_z] -> [B, 1, d_z]
        s = self.jscc_enc(z_seq, csi_vec)  # JSCC编码 [B, 1, d_s]
        z_hat_seq = self.jscc_dec(s, csi_vec)  # JSCC解码 [B, 1, d_z]
        z_hat = z_hat_seq.squeeze(1)  # [B, 1, d_z] -> [B, d_z]

        # === 解码阶段 (使用多token) ===
        decode_out = self.spi_decode(z_hat_multi, csi_vec, semantic_vec, temporal_info, z_decoded_single=z_hat)
        feat_hat = decode_out['feats_recovered']

        # 合并输出
        output = {
            **encode_out,
            **decode_out,
            'z': z,
            'z_hat': z_hat,
            'z_multi': z_multi,
            'z_hat_multi': z_hat_multi,
            's': s,
            's_multi': s_multi,
            'feat_hat': feat_hat,
        }

        return output

    def forward_spi_stage3(self, feats: torch.Tensor, csi_vec: torch.Tensor, channel_sim, snr_min_db: float, snr_max_db: float) -> Dict:
        """
        SPI版本的Stage3前向传播：Hash + JSCC + 信道噪声

        Args:
            feats: [B, T, feat_dim] 输入特征
            csi_vec: [B, d_csi] 信道状态信息
            channel_sim: 信道模拟器
            snr_min_db: 最小SNR
            snr_max_db: 最大SNR

        Returns:
            包含所有输出的字典
        """
        B, T, _ = feats.shape

        # === 编码阶段 ===
        encode_out = self.spi_encode(feats, csi_vec)
        z = encode_out['z_encoded']
        z_multi = encode_out['z_encoded_multi']  # 多token latents
        semantic_vec = encode_out['semantic_vec']
        temporal_info = encode_out['temporal_info']  # 提取时间信息

        # === Hash编码 (简化：直接对多token进行) ===
        if hasattr(self, 'hash') and self.hash is not None:
            # 对每个token进行hash (这里简化处理)
            hash_logits = self.hash.hash_encoder(z)
            hash_bits_clean = self.hash.hash_layer(hash_logits)
            jscc_input = hash_bits_clean.unsqueeze(1)  # [B, 1, hash_bits]
            jscc_input_multi = z_multi  # 直接使用多token
        else:
            jscc_input = z
            jscc_input_multi = z_multi
            hash_logits = None
            hash_bits_clean = None

        # === 多token JSCC + 信道传输 ===
        s_multi = self.jscc_enc(jscc_input_multi, csi_vec)

        # 保持单一兼容性
        s = self.jscc_enc(jscc_input.unsqueeze(1) if jscc_input.dim() == 2 else jscc_input, csi_vec)

        # 信道采样和传输
        csi_dict, amp_t, snr_db_t = channel_sim.sample_csi(
            B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
        )
        amp_t = amp_t.to(device=s_multi.device, dtype=s_multi.dtype)
        snr_db_t = snr_db_t.to(device=s_multi.device, dtype=s_multi.dtype)

        # 对多token信号应用信道噪声
        s_multi_noisy = channel_sim.apply(s_multi, amp_t, snr_db_t)
        s_noisy = channel_sim.apply(s, amp_t, snr_db_t)

        # === 多token JSCC解码 ===
        z_hat_multi = self.jscc_dec(s_multi_noisy, csi_vec)

        # === Hash解码 (单一兼容性) ===
        jscc_output = self.jscc_dec(s_noisy, csi_vec)
        if hash_bits_clean is not None:
            hash_bits_rx = self.hash.hash_layer(jscc_output)
            z_hat = self.hash.hash_decoder(hash_bits_rx)
        else:
            z_hat = jscc_output.squeeze(1) if jscc_output.dim() == 3 else jscc_output

        # === 解码阶段 (使用多token) ===
        decode_out = self.spi_decode(z_hat_multi, csi_vec, semantic_vec, temporal_info, z_decoded_single=z_hat)
        feat_hat = decode_out['feats_recovered']

        # 合并输出
        output = {
            **encode_out,
            **decode_out,
            'z': z,
            'z_hat': z_hat,
            'z_multi': z_multi,
            'z_hat_multi': z_hat_multi,
            's': s,
            's_noisy': s_noisy,
            's_multi': s_multi,
            's_multi_noisy': s_multi_noisy,
            'feat_hat': feat_hat,
            'hash_logits': hash_logits,
            'hash_bits_clean': hash_bits_clean,
            'actual_snr': 10 * torch.log10(torch.mean(s_multi.pow(2)) / (torch.mean((s_multi_noisy - s_multi).pow(2)) + 1e-8)),
        }

        return output

    def compute_spi_loss(self, spi_output: Dict, original_feats: torch.Tensor, audio_hat: torch.Tensor, audio_real: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算SPI专用损失（集成Anti-Buzz功能）

        Args:
            spi_output: SPI前向传播输出
            original_feats: 原始特征
            audio_hat: 生成音频
            audio_real: 真实音频

        Returns:
            总损失和损失字典
        """
        # 使用集成的SPILoss计算损失
        targets = {
            'original_feats': original_feats,
            'feats': original_feats
        }

        # 确保spi_output包含正确的key映射
        spi_output_mapped = spi_output.copy()
        if 'feat_hat' in spi_output:
            spi_output_mapped['feats_recovered'] = spi_output['feat_hat']

        total_loss, losses = self.spi_loss(
            spi_output_mapped,
            targets,
            audio_pred=audio_hat,
            audio_target=audio_real
        )

        return total_loss, losses

    def get_semantic_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        提取语义特征 (用于外部监督)

        Args:
            feats: [B, T, feat_dim] 输入特征

        Returns:
            semantic_vec: [B, semantic_dim] 语义向量
        """
        with torch.no_grad():
            speech_image, _ = self.speech_to_image(feats)  # 新版本返回图像和时间信息
            _, semantic_vec, _ = self.semantic_pos_encoder(speech_image)
            return semantic_vec

    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        spi_params = (
            sum(p.numel() for p in self.speech_to_image.parameters()) +
            sum(p.numel() for p in self.semantic_pos_encoder.parameters()) +
            sum(p.numel() for p in self.image_codec.parameters())
        )

        print("=" * 50)
        print("SPI-LiteSpeechJSCC Model Information")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"SPI module parameters: {spi_params:,} ({spi_params/total_params*100:.1f}%)")
        print(f"Image size: {self.img_size}x{self.img_size}")
        print(f"Latent dimension: {self.d_z}")
        print(f"Semantic dimension: {self.semantic_dim}")
        print("=" * 50)


def test_spi_lite_jscc():
    """测试SPI_LiteSpeechJSCC"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = SPI_LiteSpeechJSCC(
        feat_dim=20,
        d_csi=4,
        d_z=32,
        d_s=24,
        device=device
    ).to(device)

    model.print_model_info()

    # 测试数据
    batch_size = 2
    seq_len = 200
    feats = torch.randn(batch_size, seq_len, 20, device=device)
    csi_vec = torch.randn(batch_size, 4, device=device)

    print(f"\nTesting with input: {feats.shape}")

    # 测试Stage2
    with torch.no_grad():
        output = model.forward_spi_stage2(feats, csi_vec)
        print(f"Stage2 output keys: {list(output.keys())}")
        print(f"Reconstructed features: {output['feat_hat'].shape}")

        # 计算重建误差
        recon_error = F.mse_loss(output['feat_hat'], feats)
        print(f"Feature reconstruction error: {recon_error.item():.6f}")

    print("✓ SPI-LiteSpeechJSCC test completed successfully!")


if __name__ == "__main__":
    test_spi_lite_jscc()
