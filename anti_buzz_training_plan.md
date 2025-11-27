# Anti-Buzz训练方案：解决F0/Voicing作弊解

## 问题诊断

**当前状况：**
- F0：几乎一条直线 (mean=175.8Hz, std=0.5Hz)
- Voicing：99.8%有声，应该是32%
- 波形：振幅饱和的持续嗡声
- Mel图：整幅栅栏条纹，失去静音/共振峰结构

**根本原因：**
目标函数设计让"STFT看起来像人声"比"F0/静音结构正确"更便宜，导致网络找到一个作弊解：用恒定F0+全程有声来欺骗STFT损失。

## 解决策略：让作弊行为变得"极贵"

### Phase 1: 特征级强监督（立即实施）

#### 1.1 F0/Voicing维度加权
```python
# 从平均L1变为加权L1
loss_cep = diff[..., :18].abs().mean()           # 前18维：倒谱
loss_f0  = diff[..., 18:19].abs().mean()         # F0维
loss_vuv = diff[..., 19:20].abs().mean()         # Voicing维

loss_feat = loss_cep + 5.0 * loss_f0 + 5.0 * loss_vuv
```

#### 1.2 F0形状相关性约束
```python
# 防止只拟合均值，强制跟随F0轨迹
corr = compute_f0_correlation(feats_hat[..., 18], feats[..., 18])
loss_f0_shape = (1.0 - corr).clamp(min=0).mean()
```

#### 1.3 Voicing二分类损失
```python
# 显式的0/1监督
vuv_target = (feats[..., 19] > 0.5).float()
vuv_pred = feats_hat[..., 19].sigmoid()
loss_vuv_bce = F.binary_cross_entropy(vuv_pred, vuv_target)
```

### Phase 2: 音频级F0监督（核心武器）

#### 2.1 直接F0损失
```python
with torch.no_grad():
    f0_target = extract_f0(audio_real)    # [B, T_f0]
f0_pred = extract_f0(audio_hat)           # 同尺寸

loss_f0_audio = F.l1_loss(f0_pred, f0_target)
```

#### 2.2 F0方差下界约束
```python
# 防止F0塌缩为常数
std_target = f0_target.std(dim=1)
std_pred = f0_pred.std(dim=1)
loss_f0_variance = F.relu(20.0 - std_pred).mean()  # 至少20Hz变化
```

#### 2.3 静音段能量约束
```python
# 在静音处发出大声会被成倍惩罚
silence_mask = (energy_target < 0.01).float()
loss_silence = (wave_diff.abs() * silence_mask).mean() * 3.0
```

### Phase 3: 训练日程优化

#### 3.1 分阶段权重调整
```
Step 0-1000:   优先特征级监督，GAN权重=0
Step 1000-3000: 加入音频级F0监督，GAN权重=0.1
Step 3000-5000: 逐步恢复GAN权重到0.3
Step 5000+:    解冻FARGAN部分参数，端到端微调
```

#### 3.2 Vocoder解冻策略
```python
# 不要让vocoder永远停在"完美特征"的假设上
if step == 5000:
    # 只解冻FARGAN的高层，用小学习率
    for name, param in model.vocoder.named_parameters():
        if "layer_4" in name or "layer_5" in name:
            param.requires_grad = True
```

## 具体实施计划

### 第一周：核心损失改造
- [x] 修改SPI损失函数，实现F0/Voicing加权
- [x] 添加音频级F0提取器和损失
- [x] 实现F0方差约束和静音监督
- [x] 创建新的训练配置

### 第二周：训练验证
- [ ] 从现有checkpoint恢复，使用新损失继续训练
- [ ] 监控F0曲线是否从直线变为起伏
- [ ] 验证Mel图是否从网格变为有结构
- [ ] 对比Voicing预测准确率

### 第三周：端到端优化
- [ ] 逐步解冻FARGAN参数
- [ ] 调整各损失权重达到最优平衡
- [ ] 生成高质量音频样本验证

## 成功指标

### 定量指标：
1. **F0标准差**: 从0.5Hz提升到>20Hz
2. **Voicing准确率**: 从99.8%降到合理范围(30-70%)
3. **F0相关性**: 与真值相关性>0.7
4. **Mel MSE**: 显著下降，不再是网格状

### 定性指标：
1. **F0曲线**: 从直线变为自然起伏
2. **Mel谱图**: 出现清晰的共振峰和静音段
3. **音频质量**: 消除持续嗡声，恢复自然韵律

## 风险控制

### 潜在问题：
1. 强监督可能导致过度约束
2. 多损失权重难以平衡
3. 训练稳定性下降

### 应对措施：
1. 采用渐进式权重调整
2. 设置损失监控和早停机制
3. 保留多个checkpoint便于回退

## 预期效果

通过这套"反作弊"策略，预期在1-2周内：
- **消除嗡嗡声**: F0恢复自然变化
- **恢复静音**: Voicing预测变准确
- **改善结构**: Mel图显示正确的时频结构
- **提升质量**: 音频恢复自然语音特性

这是一个针对性极强的修复方案，直接打击导致当前问题的根本原因。