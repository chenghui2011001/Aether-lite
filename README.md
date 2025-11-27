# Aether-lite: SPI-JSCC Speech Compression

## Overview

Aether-lite implements a lightweight Speech-to-Image Joint Source-Channel Coding (SPI-JSCC) system for ultra-low bitrate speech compression and transmission. The system features Anti-Buzz training to fix F0/Voicing collapse issues and supports distributed multi-GPU training.

## Key Features

- **SPI-JSCC Architecture**: Speech-to-Image mapping with joint source-channel coding
- **Anti-Buzz Training**: Progressive 4-stage training to fix F0/Voicing collapse
- **Multi-GPU Support**: Distributed training with DistributedDataParallel
- **YIN F0 Extraction**: Industry-standard F0 extraction for reliable pitch tracking
- **Progressive Training**: Feature focus → Audio supervision → GAN recovery → End-to-end

## Quick Start

### Single GPU Training
```bash
python training/spi_jscc_train.py \
    --save_visualization \
    --use_stft_loss \
    --fargan_ckpt ./fargan_pt/fargan_sq1Ab_adv_50.pth \
    --data_root ./data_cn \
    --lambda_wave 0.1 \
    --lambda_adv 0.0 \
    --lambda_spi 1.5
```

### Multi-GPU Training
```bash
# 2 GPUs
torchrun --nproc_per_node=2 training/spi_jscc_train.py \
    --distributed \
    --save_visualization \
    --use_stft_loss \
    --fargan_ckpt ./fargan_pt/fargan_sq1Ab_adv_50.pth \
    --data_root ./data_cn \
    --lambda_wave 0.1 \
    --lambda_adv 0.0 \
    --lambda_spi 1.5

# 4 GPUs
torchrun --nproc_per_node=4 training/spi_jscc_train.py \
    --distributed \
    --save_visualization \
    --use_stft_loss \
    --fargan_ckpt ./fargan_pt/fargan_sq1Ab_adv_50.pth \
    --data_root ./data_cn \
    --lambda_wave 0.1 \
    --lambda_adv 0.0 \
    --lambda_spi 1.5
```

## Architecture

```
Speech Input → SPI Encoder → JSCC → Channel → JSCC Decoder → SPI Decoder → FARGAN Vocoder → Audio Output
                     ↓                                           ↑
              Semantic Features                           Feature Reconstruction
                     ↓                                           ↑
              Anti-Buzz Supervision ←→ YIN F0 Extractor → Audio-level F0 Loss
```

## Anti-Buzz Training Strategy

The system uses a progressive 4-stage training approach to eliminate the "buzz artifact" problem:

1. **Stage 1 (0-1000 steps)**: Feature-level supervision, GAN disabled
2. **Stage 2 (1000-3000 steps)**: Audio-level supervision, weak GAN
3. **Stage 3 (3000-5000 steps)**: Progressive GAN weight recovery
4. **Stage 4 (5000+ steps)**: FARGAN unfreeze, end-to-end optimization

## Dependencies

- PyTorch >= 1.12
- torchaudio
- numpy
- tqdm
- librosa (for YIN F0 extraction)

## Data Format

- Features: `.f32` files (36-dimensional FARGAN features)
- Audio: `.pcm` files (16kHz, 16-bit)

## Model Components

- **SPI Encoder**: Speech → Image latent mapping
- **JSCC Codec**: Joint source-channel coding for robust transmission
- **Hash Bottleneck**: Optional discrete representation
- **FARGAN Vocoder**: High-quality neural vocoder

## Training Progress

The training uses tqdm progress bars showing:
- Current epoch/step
- Training stage (feature_focus, audio_weak_gan, progressive_gan, end_to_end)
- Loss components (total, recon, adv, spi)
- FARGAN status (frozen/unfrozen)

## License

This project follows the same license as the parent FARGAN project.