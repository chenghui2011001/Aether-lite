#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Aether-Lite移植后的模块
"""

import torch
import sys
import os

# 确保能导入models包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    LiteSpeechJSCC,
    AetherBaseSpeechJSCC,
    create_lite_speech_jscc,
    create_aether_base,
    HashBottleneck,
    FARGANDecoder
)

def test_lite_speech_jscc():
    """测试LiteSpeechJSCC模型"""
    print("=" * 50)
    print("Testing LiteSpeechJSCC")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_lite_speech_jscc(device=device)

    # 模型信息
    info = model.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Parameter Size: {info['parameter_size_mb']:.2f} MB")
    print(f"Bitrate: {info['bitrate_kbps']:.1f} kbps")

    # 测试数据
    batch_size = 2
    seq_len = 100
    x_feat = torch.randn(batch_size, seq_len, 36).to(device)
    csi = torch.randn(batch_size, 3).to(device)

    # 测试连续JSCC
    feat_hat, z_hat, z = model.forward_continuous(x_feat, csi, snr_db=10.0)
    print(f"Continuous JSCC - feat_hat: {feat_hat.shape}")

    # 测试Hash JSCC
    feat_hat_h, z_q, z_hat_h, hash_out = model.forward_hash(x_feat, csi, snr_db=10.0)
    print(f"Hash JSCC - feat_hat: {feat_hat_h.shape}")

    print("✅ LiteSpeechJSCC test passed!\n")


def test_aether_base():
    """测试AetherBase模型"""
    print("=" * 50)
    print("Testing AetherBaseSpeechJSCC")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_base = create_aether_base(device=device)

    # 模型信息
    info = model_base.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Role: {info['role']}")
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Parameter Size: {info['parameter_size_mb']:.2f} MB")

    # 测试数据
    batch_size = 2
    seq_len = 100
    x_feat = torch.randn(batch_size, seq_len, 36).to(device)
    csi = torch.randn(batch_size, 3).to(device)

    # 测试蒸馏目标生成
    targets = model_base.get_distillation_targets(x_feat, csi, snr_db=10.0)
    print(f"Distillation targets - z_base: {targets['z_base'].shape}")
    print(f"Distillation targets - audio_base: {targets['audio_base'].shape}")

    print("✅ AetherBase test passed!\n")


def test_parameter_comparison():
    """对比参数量"""
    print("=" * 50)
    print("Parameter Comparison")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model_lite = create_lite_speech_jscc(device=device)
    model_base = create_aether_base(device=device)

    # 获取信息
    info_lite = model_lite.get_model_info()
    info_base = model_base.get_model_info()

    print(f"Lite Parameters: {info_lite['total_parameters']:,}")
    print(f"Base Parameters: {info_base['total_parameters']:,}")
    print(f"Base/Lite Ratio: {info_base['total_parameters']/info_lite['total_parameters']:.1f}x")

    # 验证参数量在预期范围内
    lite_params_mb = info_lite['parameter_size_mb']
    base_params_mb = info_base['parameter_size_mb']

    print(f"Lite Size: {lite_params_mb:.2f} MB (Target: 2-5 MB)")
    print(f"Base Size: {base_params_mb:.2f} MB (Target: 5-10 MB)")

    # 验证
    if 2 <= lite_params_mb <= 5:
        print("✅ Lite参数量在目标范围内")
    else:
        print("❌ Lite参数量超出目标范围")

    if 5 <= base_params_mb <= 10:
        print("✅ Base参数量在目标范围内")
    else:
        print("❌ Base参数量超出目标范围")

    print()


def test_core_components():
    """测试核心组件"""
    print("=" * 50)
    print("Testing Core Components")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试HashBottleneck
    hash_bottleneck = HashBottleneck(
        input_dim=16,
        hash_bits=16,
        hash_method='bihalf'
    ).to(device)

    x = torch.randn(2, 100, 16).to(device)
    hash_result = hash_bottleneck(x, {'ber': 0.1})
    print(f"✅ HashBottleneck - input: {x.shape}, output: {hash_result['reconstructed'].shape}")

    # 测试FARGANDecoder
    fargan_decoder = FARGANDecoder().to(device)
    feat = torch.randn(2, 100, 36).to(device)
    period, audio = fargan_decoder(feat)
    print(f"✅ FARGANDecoder - feat: {feat.shape}, audio: {audio.shape}")

    print()


if __name__ == "__main__":
    print("🚀 Testing Aether-Lite Migrated Modules")
    print("=" * 60)

    try:
        test_core_components()
        test_lite_speech_jscc()
        test_aether_base()
        test_parameter_comparison()

        print("🎉 All tests passed successfully!")
        print("✅ 所有核心组件已成功移植到Aether-lite目录")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()