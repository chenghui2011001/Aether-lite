#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lite AETHER JSCC training pipeline (Stage 2 / Stage 3).

设计目标：
- 复用 Aether-lite 中已经实现好的组件：
  - 数据加载：utils.real_data_loader.AETHERRealDataset / create_combined_data_loader
  - 模型：LiteSpeechJSCC / HashBottleneck / FARGANDecoder / ChannelSimulator
  - 损失：fargan_wave_losses / compute_layered_loss / HashBottleneck 正则
- 提供一个清晰的端到端训练通路，与需求中描述的结构对齐：

  输入: Features[B,T,36], Audio[B,T*160], CSI[B,4]

  === 编码端 ===
  Features[B,T,36] + CSI[B,4]
      ↓ EncoderLite (encoder)
  Latent z[B,T,d_z]
      ↓ [Stage 3+] HashEncoder (hash_bottleneck.hash_encoder)
      ↓ [Stage 3+] Hash logits → Hash bits
      ↓ JSCCEncoder (jscc_encoder) + CSI[B,4]
  Symbols s[B,T,d_s]

  === 信道传输 ===
  Symbols s[B,T,d_s]
      ↓ ChannelSimulator.sample_csi + ChannelSimulator.apply
  s_noisy[B,T,d_s]

  === 解码端 ===
  s_noisy[B,T,d_s] + CSI[B,4]
      ↓ JSCCDecoder (jscc_decoder)
      ↓ [Stage 3+] Hash bits (soft → hard)
      ↓ [Stage 3+] HashDecoder (hash_bottleneck.hash_decoder)
  Reconstructed z_hat[B,T,d_z]
      ↓ DecoderLite (decoder) + CSI[B,4]
  Features_pred[B,T,36]
      ↓ VocoderLite (FARGANDecoder)
  Audio_pred[B,T*160]

说明：
- 为了避免破坏现有接口，LiteSpeechJSCC 本身不强制修改 forward；
  训练脚本直接调用其子模块 encoder/jscc/hash/decoder/vocoder 组成训练前向。
- Stage2（连续 JSCC）只使用 encoder/jscc/decoder/vocoder；
  Stage3 在此基础上插入 HashEncoder/Decoder 与 bit-level 正则。
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
import torchaudio
import numpy as np

from utils.multi_stft_discriminator import create_adversarial_wave_loss
from models import LiteSpeechJSCC
from models.semantic_extractor import create_semantic_extractor
from utils.channel_sim import ChannelSimulator
from utils.real_data_loader import create_combined_data_loader
from training.fargan_losses import multi_resolution_stft_loss


@dataclass
class LiteTrainConfig:
    """简单训练配置（可根据需要扩展/外部化到 YAML）。"""

    data_root: str = "/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/data_expert_augmented_small200k"
    stage: int = 2  # 2: 连续JSCC; 3: Hash + JSCC
    batch_size: int = 8
    sequence_length: int = 200  # 帧数；200帧≈2s
    num_epochs: int = 1         # 演示用途，实际训练请增大
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查点保存
    output_dir: str = "./checkpoints_lite_jscc"
    save_every_steps: int = 500
    resume_from: Optional[str] = None  # 恢复训练的检查点路径

    # 音频验证保存
    save_audio: bool = False           # 是否保存验证音频
    audio_save_dir: str = "./audio_samples"  # 音频保存目录
    save_audio_every_steps: int = 1000  # 每N步保存一次音频
    max_audio_samples: int = 4         # 每次最多保存几个音频样本

    # 损失权重
    lambda_wave: float = 1.0    # 时域波形损失权重 (L1 或 STFT)
    lambda_stft: float = 1.0    # 多分辨率STFT损失权重
    lambda_adv: float = 0.3     # 对抗波形损失权重
    lambda_hash_recon: float = 0.1
    lambda_hash_reg: float = 0.1

    # 损失类型选择
    use_stft_loss: bool = True  # 使用STFT损失替代L1

    # 通道配置
    snr_min_db: float = -5.0
    snr_max_db: float = 15.0


def csi_dict_to_vec(csi: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    将 ChannelSimulator.sample_csi 的输出字典打包为 CSI[B,4] 向量：
        [snr_proxy, time_selectivity, freq_selectivity, los_ratio]
    """
    return torch.stack(
        [
            csi["snr_proxy"],
            csi["time_selectivity"],
            csi["freq_selectivity"],
            csi["los_ratio"],
        ],
        dim=-1,
    )


def save_checkpoint(
    model: LiteSpeechJSCC,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    cfg: LiteTrainConfig,
    epoch: int,
    global_step: int,
    output_dir: str,
) -> None:
    """保存模型检查点。"""
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step:06d}.pth")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "disc_optimizer_state_dict": disc_optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": {
            "stage": cfg.stage,
            "batch_size": cfg.batch_size,
            "sequence_length": cfg.sequence_length,
            "lr": cfg.lr,
            "lambda_wave": cfg.lambda_wave,
            "lambda_adv": cfg.lambda_adv,
            "lambda_hash_recon": cfg.lambda_hash_recon,
            "lambda_hash_reg": cfg.lambda_hash_reg,
            "snr_min_db": cfg.snr_min_db,
            "snr_max_db": cfg.snr_max_db,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: LiteSpeechJSCC,
    optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    """
    加载检查点并恢复模型和优化器状态。

    Returns:
        Tuple[int, int]: (epoch, global_step) 恢复的训练状态
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型状态
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])

    # 恢复训练状态
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    print(f"Resumed from epoch {epoch}, step {global_step}")
    print(f"Training config from checkpoint:")
    for key, value in checkpoint.get("config", {}).items():
        print(f"  {key}: {value}")

    return epoch, global_step


def save_audio_samples(
    audio_real: torch.Tensor,
    audio_gen: torch.Tensor,
    cfg: LiteTrainConfig,
    epoch: int,
    global_step: int,
    sample_rate: int = 16000,
) -> None:
    """
    保存验证音频样本用于质量监控。

    Args:
        audio_real: 真实音频 [B, L]
        audio_gen: 生成音频 [B, L]
        cfg: 训练配置
        epoch: 当前epoch
        global_step: 当前step
        sample_rate: 采样率
    """
    if not cfg.save_audio:
        return

    os.makedirs(cfg.audio_save_dir, exist_ok=True)

    # 限制保存的样本数量
    batch_size = min(audio_real.size(0), cfg.max_audio_samples)

    for i in range(batch_size):
        # 获取音频样本并转换为CPU
        real_sample = audio_real[i].detach().cpu()
        gen_sample = audio_gen[i].detach().cpu()

        # 确保音频长度一致
        min_len = min(real_sample.size(0), gen_sample.size(0))
        real_sample = real_sample[:min_len]
        gen_sample = gen_sample[:min_len]

        # 归一化到 [-1, 1] 范围
        real_sample = real_sample / (real_sample.abs().max() + 1e-8)
        gen_sample = gen_sample / (gen_sample.abs().max() + 1e-8)

        # 保存真实音频
        real_path = os.path.join(
            cfg.audio_save_dir,
            f"step_{global_step:06d}_sample_{i:02d}_real.wav"
        )
        torchaudio.save(real_path, real_sample.unsqueeze(0), sample_rate)

        # 保存生成音频
        gen_path = os.path.join(
            cfg.audio_save_dir,
            f"step_{global_step:06d}_sample_{i:02d}_gen.wav"
        )
        torchaudio.save(gen_path, gen_sample.unsqueeze(0), sample_rate)

    print(f"Audio samples saved: {batch_size} pairs at step {global_step}")


def forward_stage2(
    model: LiteSpeechJSCC,
    feats: torch.Tensor,
    audio: torch.Tensor,
    channel_sim: ChannelSimulator,
    snr_min_db: float,
    snr_max_db: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Stage2: 连续 JSCC + FARGAN vocoder 前向，不含 Hash。

    只做前向，不在此处计算损失；由外层训练循环决定如何组合 AdversarialWaveLoss 等。
    """
    B, T, _ = feats.shape

    # === 信道采样 & CSI 聚合 ===
    csi_dict, amp_t, snr_db_t = channel_sim.sample_csi(
        B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
    )
    csi_vec_model = csi_dict_to_vec(csi_dict).to(device=device, dtype=feats.dtype)  # [B,4]

    # === 编码端 ===
    z = model.enc(feats, csi_vec_model)                    # [B,T,d_z]
    s = model.jscc_enc(z, csi_vec_model)                   # [B,T,d_s]

    # === 信道传输 ===
    amp_t = amp_t.to(device=s.device, dtype=s.dtype)
    snr_db_t = snr_db_t.to(device=s.device, dtype=s.dtype)
    s_noisy = channel_sim.apply(s, amp_t, snr_db_t)        # [B,T,d_s]

    # === 解码端 ===
    z_hat = model.jscc_dec(s_noisy, csi_vec_model, h_rayleigh=None)  # [B,T,d_z]
    feat_hat = model.dec(z_hat, csi_vec_model)             # [B,T,36]

    target_len = audio.size(-1)
    period, audio_hat = model.vocoder(feat_hat, target_len=target_len)  # period[B,T'], audio_hat[B,1,L]
    audio_hat = audio_hat.squeeze(1)                       # [B,L]
    # 确保生成音频与目标音频长度完全一致
    if audio_hat.size(-1) != audio.size(-1):
        min_len = min(audio_hat.size(-1), audio.size(-1))
        audio_hat = audio_hat[..., :min_len]
        audio = audio[..., :min_len]  

    return {
        "audio_hat": audio_hat,
        "audio": audio,
        "period": period,
        "feat_hat": feat_hat,
        "z": z,
        "z_hat": z_hat,
        "csi_vec": csi_vec_model,
    }


def forward_stage3(
    model: LiteSpeechJSCC,
    feats: torch.Tensor,
    audio: torch.Tensor,
    channel_sim: ChannelSimulator,
    snr_min_db: float,
    snr_max_db: float,
    cfg: LiteTrainConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Stage3: Hash + bit-level JSCC + FARGAN vocoder 前向。

    约定：model.hash.hash_bits == model.d_z，使 Hash bits 维度与 JSCC latent 维度一致，
    方便直接作为 JSCCEncoder 的输入 / JSCCDecoder 的输出。
    """
    B, T, _ = feats.shape

    # === 信道采样 & CSI 聚合 ===
    csi_dict, amp_t, snr_db_t = channel_sim.sample_csi(
        B, T, channel="fading", snr_min_db=snr_min_db, snr_max_db=snr_max_db
    )
    csi_vec_model = csi_dict_to_vec(csi_dict).to(device=device, dtype=feats.dtype)  # [B,4]

    # === 编码端 ===
    z = model.enc(feats, csi_vec_model)                    # [B,T,d_z]

    # HashEncoder: 连续 latent → hash logits → hash bits
    hash_logits = model.hash.hash_encoder(z)               # [B,T,K]
    hash_bits_clean = model.hash.hash_layer(hash_logits)   # [B,T,K] ∈{-1,+1}

    # JSCCEncoder: bit 序列 + CSI → 模拟码 symbol
    s = model.jscc_enc(hash_bits_clean, csi_vec_model)     # [B,T,d_s]

    # === 信道传输 ===
    amp_t = amp_t.to(device=s.device, dtype=s.dtype)
    snr_db_t = snr_db_t.to(device=s.device, dtype=s.dtype)
    s_noisy = channel_sim.apply(s, amp_t, snr_db_t)        # [B,T,d_s]

    # === 解码端 ===
    # JSCCDecoder 输出“软 bit logits”
    hash_logits_rx = model.jscc_dec(s_noisy, csi_vec_model, h_rayleigh=None)  # [B,T,K]

    # 软 → 硬 bit（与编码端相同的 hash_layer）
    hash_bits_rx = model.hash.hash_layer(hash_logits_rx)   # [B,T,K]

    # HashDecoder: bit → 连续 latent
    z_hat = model.hash.hash_decoder(hash_bits_rx)          # [B,T,d_z]

    # DecoderLite: latent + CSI → 36D 特征
    feat_hat = model.dec(z_hat, csi_vec_model)             # [B,T,36]

    # Vocoder: 36D 特征 → 波形
    target_len = audio.size(-1)
    period, audio_hat = model.vocoder(feat_hat, target_len=target_len)  # [B,T'], [B,1,L]
    audio_hat = audio_hat.squeeze(1)                       # [B,L]

    return {
        "audio_hat": audio_hat,
        "audio": audio,
        "period": period,
        "feat_hat": feat_hat,
        "z": z,
        "z_hat": z_hat,
        "hash_logits": hash_logits,
        "hash_bits_clean": hash_bits_clean,
        "csi_vec": csi_vec_model,
    }


def build_dataloader(cfg: LiteTrainConfig):
    """
    创建 AETHER 数据加载器，直接使用 36D FARGAN 特征规范。
    """
    dataloader, _dataset = create_combined_data_loader(
        data_root=cfg.data_root,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        max_samples=None,
        num_workers=4,
        energy_selection=True,
    )
    return dataloader


def build_model(cfg: LiteTrainConfig) -> LiteSpeechJSCC:
    """
    构造 LiteSpeechJSCC 模型实例。

    注意：这里使用 d_csi=4，与训练通路中的 CSI[B,4] 一致。
    """
    device = torch.device(cfg.device)
    model = LiteSpeechJSCC(
        feat_dim=36,
        d_csi=4,
        d_z=16,
        d_s=16,
        n_bits=16,
        hidden=80,
        hash_method="bihalf",
        device=device,
    )
    return model


def train(cfg: LiteTrainConfig):
    device = torch.device(cfg.device)
    dataloader = build_dataloader(cfg)
    model = build_model(cfg)
    model.to(device)

    # HuBERT 语义特征提取器：输出 16 维语义向量，与 20 维声学特征拼接
    semantic_extractor = create_semantic_extractor(
        model_name="hubert-base",
        proj_dim=16,
        device=device,
    )
    semantic_extractor.eval()
    for p in semantic_extractor.parameters():
        p.requires_grad_(False)

    channel_sim = ChannelSimulator(sample_rate=16000, frame_hz=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Adversarial wave loss & discriminator optimizer
    adv_wave_loss = create_adversarial_wave_loss(
        fft_sizes=[512, 256],
        hop_factors=4,
        base_channels=16,
        feature_match_weight=5.0,
        adversarial_weight=1.0,
    ).to(device)
    disc_optimizer = torch.optim.AdamW(
        adv_wave_loss.get_discriminator_parameters(),
        lr=1e-4, betas=(0.5, 0.9)
    )

    # 恢复检查点（如果指定）
    start_epoch = 0
    global_step = 0
    if cfg.resume_from:
        start_epoch, global_step = load_checkpoint(
            cfg.resume_from, model, optimizer, disc_optimizer, device
        )
        # 如果从检查点恢复，需要调整起始epoch
        start_epoch = start_epoch if global_step > 0 else 0

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        for batch in dataloader:
            feats_raw = batch["x"].to(device)      # [B,T,36]
            audio = batch["audio"].to(device)  # [B,L]

            B, T, _ = feats_raw.shape

            # 20 维声学特征：来自 f32 前 20 维
            feats_acoustic = feats_raw[..., :20]  # [B,T,20]

            # 16 维语义特征：HuBERT-based semantic extractor（冻结，仅作 teacher）
            with torch.no_grad():
                sem_16 = semantic_extractor(audio, target_frames=T)  # [B,T,16]

            feats = torch.cat([feats_acoustic, sem_16], dim=-1)  # [B,T,36]

            # === 前向 ===
            if cfg.stage == 2:
                out = forward_stage2(
                    model=model,
                    feats=feats,
                    audio=audio,
                    channel_sim=channel_sim,
                    snr_min_db=cfg.snr_min_db,
                    snr_max_db=cfg.snr_max_db,
                    device=device,
                )
            else:
                out = forward_stage3(
                    model=model,
                    feats=feats,
                    audio=audio,
                    channel_sim=channel_sim,
                    snr_min_db=cfg.snr_min_db,
                    snr_max_db=cfg.snr_max_db,
                    cfg=cfg,
                    device=device,
                )

            audio_hat = out["audio_hat"]
            audio_real = out["audio"]

            # === 判别器步骤 (D-step) ===
            disc_optimizer.zero_grad(set_to_none=True)
            disc_out = adv_wave_loss.discriminator_step(
                audio_real=audio_real,
                audio_gen=audio_hat.detach(),
            )
            loss_d = disc_out["discriminator_loss"]
            loss_d.backward()
            if global_step % 2 == 0:
                disc_optimizer.step()

            # === 生成器步骤 (G-step) ===
            optimizer.zero_grad(set_to_none=True)

            gen_out = adv_wave_loss.generator_step(
                audio_real=audio_real,
                audio_gen=audio_hat,
            )
            loss_adv = gen_out["total_adversarial_loss"]

            # 音频重建损失 - 可选L1或STFT
            if cfg.use_stft_loss:
                # 多分辨率STFT损失：更好的频域感知
                loss_stft = multi_resolution_stft_loss(
                    audio_hat, audio_real,
                    device=device,
                    fft_sizes=[1024, 512, 256, 128],
                    hop_sizes=[256, 128, 64, 32],
                    win_lengths=[1024, 512, 256, 128]
                )
                loss_recon = loss_stft
                loss_type = "stft"
            else:
                # 传统L1损失
                loss_recon = F.l1_loss(audio_hat, audio_real)
                loss_type = "l1"

            # 自动缩放对抗损失，防止其主导优化
            eps = 1e-6
            adv_scale = (loss_recon.detach() + eps) / (loss_adv.detach() + eps)
            adv_scale = torch.clamp(adv_scale, 0.1, 1.0)  # 限制缩放范围

            total = cfg.lambda_wave * loss_recon + cfg.lambda_adv * adv_scale * loss_adv

            # Stage3: 叠加 Hash 相关损失
            if cfg.stage == 3:
                z = out["z"]
                z_hat = out["z_hat"]
                hash_logits = out["hash_logits"]
                hash_bits_clean = out["hash_bits_clean"]

                hash_recon = F.mse_loss(z_hat, z.detach())
                hash_reg_terms = model.hash.compute_hash_regularization(hash_logits, hash_bits_clean)
                hash_reg = sum(hash_reg_terms.values())

                total = (
                    total
                    + cfg.lambda_hash_recon * hash_recon
                    + cfg.lambda_hash_reg * hash_reg
                )

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if global_step % 10 == 0:
                log = (
                    f"[epoch {epoch} step {global_step}] "
                    f"total={total.item():.4f} | "
                    f"recon({loss_type})={loss_recon.item():.4f} | "
                    f"adv={loss_adv.item():.4f} (scale={adv_scale.item():.3f}) | "
                    f"d_loss={loss_d.item():.4f}"
                )
                if cfg.stage == 3:
                    log += f" | hash_recon={hash_recon.item():.4f} hash_reg={hash_reg.item():.4f}"
                print(log)

            # Save checkpoint every save_every_steps
            if global_step % cfg.save_every_steps == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    disc_optimizer=disc_optimizer,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    output_dir=cfg.output_dir,
                )

            # Save audio samples for validation
            if cfg.save_audio and global_step % cfg.save_audio_every_steps == 0:
                save_audio_samples(
                    audio_real=audio_real,
                    audio_gen=audio_hat,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                )

            global_step += 1


def parse_args() -> LiteTrainConfig:
    parser = argparse.ArgumentParser(description="Lite AETHER JSCC training (Stage2/3)")
    parser.add_argument("--data_root", type=str, default="/home/bluestar/FARGAN/opus/dnn/torch/Aether-lite/data_expert_augmented_small200k")
    parser.add_argument("--stage", type=int, choices=[2, 3], default=2)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=600)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lite_jscc", help="Directory to save checkpoints")
    parser.add_argument("--save_every_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint file to resume from")

    # Audio validation arguments
    parser.add_argument("--save_audio", action="store_true", help="Enable audio sample saving for validation")
    parser.add_argument("--audio_save_dir", type=str, default="./audio_samples", help="Directory to save audio samples")
    parser.add_argument("--save_audio_every_steps", type=int, default=50, help="Save audio samples every N steps")
    parser.add_argument("--max_audio_samples", type=int, default=4, help="Maximum number of audio samples to save per step")

    # Loss function arguments
    parser.add_argument("--use_stft_loss", action="store_true", default=True, help="Use multi-resolution STFT loss instead of L1")
    parser.add_argument("--use_l1_loss", dest="use_stft_loss", action="store_false", help="Use L1 loss instead of STFT loss")
    parser.add_argument("--lambda_wave", type=float, default=1.0, help="Weight for waveform reconstruction loss")

    parser.add_argument("--snr_min_db",type=int,default=-5)
    parser.add_argument("--snr_max_db",type=int,default=0)
    args = parser.parse_args()
    return LiteTrainConfig(
        data_root=args.data_root,
        stage=args.stage,
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
        use_stft_loss=args.use_stft_loss,
        lambda_wave=args.lambda_wave,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
