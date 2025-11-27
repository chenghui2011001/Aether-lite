"""
Aether-Lite Models

Lyra2风格轻量级语音JSCC系统的模型组件
"""

from .lite_speech_jscc import (
    LiteSpeechJSCC,
    EncoderLite,
    JSCCEncoder,
    JSCCDecoder,
    DecoderLite,
    create_lite_speech_jscc
)

from .aether_base import (
    AetherBaseSpeechJSCC,
    create_aether_base
)

# 复用的核心组件
from .hash_bottleneck import HashBottleneck
from .fargan_decoder import FARGANDecoder
from .semantic_extractor import SemanticFeatureExtractor, create_semantic_extractor
from .stablecodec_teacher import StableCodecTeacher

# SPI (语义映射+位置编码+图像编码) 模块
from .spi_modules import (
    SpeechToImageTransform,
    SemanticPositionalEncoder,
    LightweightImageCodec,
    SPILoss
)

from .spi_lite_jscc import SPI_LiteSpeechJSCC

__all__ = [
    # Lite模型
    'LiteSpeechJSCC',
    'EncoderLite',
    'JSCCEncoder',
    'JSCCDecoder',
    'DecoderLite',
    'create_lite_speech_jscc',

    # Base模型
    'AetherBaseSpeechJSCC',
    'create_aether_base',

    # SPI模型 (创新架构)
    'SPI_LiteSpeechJSCC',
    'SpeechToImageTransform',
    'SemanticPositionalEncoder',
    'LightweightImageCodec',
    'SPILoss',

    # 核心组件
    'HashBottleneck',
    'FARGANDecoder',
    'SemanticFeatureExtractor',
    'create_semantic_extractor',
    'StableCodecTeacher'
]