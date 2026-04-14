# 退化通道技术文档

## 1. 概述

退化通道（Degradation Channels）是图像修复评估系统中的重要组成部分，用于模拟图像在各种情况下的退化状态。通过应用不同类型的退化，可以评估修复算法在各种退化条件下的表现。

## 2. 核心概念

### 2.1 通道基类

所有退化通道都继承自 `BaseDegradationChannel` 基类，该基类定义了统一的接口：

```python
class BaseDegradationChannel:
    def __init__(self, **params):
        self.params = params
    
    def process(self, image):
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return image
```

### 2.2 通道工作流程

1. **初始化**：创建通道实例时设置退化参数
2. **处理**：调用 `process()` 方法对输入图像应用退化
3. **输出**：返回退化后的图像

## 3. 已有退化通道

### 3.1 高斯噪声通道

**类名**：`GaussianNoiseChannel`
**功能**：添加高斯分布的随机噪声
**参数**：
- `mean`：噪声均值，默认 0
- `sigma`：噪声标准差，默认 25

**效果**：模拟传感器噪声、低光拍摄等场景

### 3.2 椒盐噪声通道

**类名**：`SaltAndPepperChannel`
**功能**：添加盐（白色）和椒（黑色）噪声
**参数**：
- `salt_prob`：盐噪声概率，默认 0.02
- `pepper_prob`：椒噪声概率，默认 0.02

**效果**：模拟传输错误、传感器故障等场景

### 3.3 运动模糊通道

**类名**：`MotionBlurChannel`
**功能**：模拟相机或物体运动导致的模糊
**参数**：
- `kernel_size`：模糊核大小，默认 15
- `angle`：模糊方向（角度），默认 45
- `sigma`：高斯运动核的标准差，默认 2

**效果**：模拟手持拍摄、移动物体等场景

### 3.4 高斯模糊通道

**类名**：`GaussianBlurChannel`
**功能**：应用高斯模糊
**参数**：
- `kernel_size`：模糊核大小，默认 (5, 5)
- `sigma`：高斯标准差，默认 0

**效果**：模拟失焦、大气散射等场景

### 3.5 散焦模糊通道

**类名**：`DefocusBlurChannel`
**功能**：模拟镜头散焦效果
**参数**：
- `radius`：圆盘半径（2~8），默认 5

**效果**：模拟光学失焦、景深效果等场景

### 3.6 下采样通道

**类名**：`DownsampleChannel`
**功能**：降低图像分辨率
**参数**：
- `scale`：缩放比例，默认 0.5

**效果**：模拟低分辨率图像、压缩 artifacts 等场景

### 3.7 复合退化通道

**类名**：`CompositeDegradationChannel`
**功能**：顺序应用多个退化通道
**参数**：
- `channels`：退化通道列表

**方法**：
- `add_channel(channel)`：添加退化通道
- `process(image)`：按顺序应用所有通道
- `get_channel_names()`：获取通道名称列表

**效果**：模拟多种退化叠加的复杂场景

## 4. 效果对比

| 通道类型 | 适用场景 | 视觉效果 | 对修复的挑战 |
|---------|---------|---------|------------|
| 高斯噪声 | 传感器噪声、低光 | 均匀的噪点 | 去噪的同时保留细节 |
| 椒盐噪声 | 传输错误、传感器故障 | 随机的黑白噪点 | 去除噪声而不影响周围像素 |
| 运动模糊 | 运动物体、手持拍摄 | 方向性模糊 | 恢复清晰边缘和细节 |
| 高斯模糊 | 失焦、大气散射 | 均匀模糊 | 增强边缘和细节 |
| 散焦模糊 | 光学失焦、景深 | 柔和的模糊 | 恢复清晰的焦点区域 |
| 下采样 | 低分辨率、压缩 | 像素化、细节丢失 | 超分辨率重建 |

## 5. 创建新的退化通道

### 5.1 步骤

1. **继承基类**：创建一个继承自 `BaseDegradationChannel` 的新类
2. **实现构造函数**：初始化退化参数
3. **重写 process 方法**：实现退化逻辑
4. **更新导入**：在 `channels/__init__.py` 中添加新通道

### 5.2 示例：创建新的退化通道

```python
# custom_degradation.py
import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class CustomDegradationChannel(BaseDegradationChannel):
    def __init__(self, param1=1, param2=2):
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2
    
    def process(self, image):
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 实现自定义退化逻辑
        # ...
        
        return degraded_image

# 在 channels/__init__.py 中添加
from .custom_degradation import CustomDegradationChannel

__all__ = [
    # 其他通道...
    'CustomDegradationChannel'
]
```

### 5.3 最佳实践

- **参数验证**：验证输入参数的有效性
- **边界处理**：处理图像边界情况
- **性能考虑**：优化计算密集型操作
- **文档**：为新通道添加详细的文档注释
- **测试**：为新通道编写测试用例

## 6. 使用示例

### 6.1 基本使用

```python
from channels import GaussianNoiseChannel, MotionBlurChannel

# 创建退化通道
noise_channel = GaussianNoiseChannel(sigma=20)
motion_channel = MotionBlurChannel(kernel_size=12, angle=30)

# 应用退化
image = load_image()
degraded1 = noise_channel.process(image)  # 只添加噪声
degraded2 = motion_channel.process(image)  # 只添加运动模糊
```

### 6.2 使用复合通道

```python
from channels import (
    GaussianNoiseChannel,
    GaussianBlurChannel,
    CompositeDegradationChannel
)

# 创建复合通道
composite_channel = CompositeDegradationChannel()
composite_channel.add_channel(GaussianNoiseChannel(sigma=15))
composite_channel.add_channel(GaussianBlurChannel(sigma=2))

# 应用复合退化
degraded = composite_channel.process(image)
```

### 6.3 与评估系统集成

```python
from core import ChannelEvaluator

# 使用字符串标签创建通道
evaluator = ChannelEvaluator()
evaluator.evaluate(
    degradation_type='motion_blur',
    restoration_type='richardson_lucy',
    degradation_params={'kernel_size': 15, 'angle': 45},
    num_tests=5
)

# 或直接使用通道对象
from core import BaseEvaluator
from channels import MotionBlurChannel, RichardsonLucyChannel, RecognitionChannel

degradation_channel = MotionBlurChannel(kernel_size=15, angle=45)
restoration_channel = RichardsonLucyChannel(iterations=30)
recognition_channel = RecognitionChannel()

evaluator = BaseEvaluator()
evaluator.evaluate(degradation_channel, restoration_channel, recognition_channel, num_tests=5)
```

## 7. 总结

退化通道系统提供了一个灵活、可扩展的框架，用于模拟各种图像退化场景。通过使用不同的退化通道，可以全面评估修复算法在各种条件下的性能。同时，该系统的模块化设计使得添加新的退化类型变得简单直接，为未来的扩展提供了良好的基础。
