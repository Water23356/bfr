# 修复通道技术文档

## 1. 概述

修复通道（Restoration Channels）是图像修复评估系统中的核心组件，用于对退化图像进行修复处理。通过应用不同类型的修复算法，可以评估各种修复方法在不同退化场景下的效果。

## 2. 核心概念

### 2.1 通道基类

所有修复通道都继承自 `BaseRestorationChannel` 基类，该基类定义了统一的接口：

```python
class BaseRestorationChannel:
    def __init__(self, **params):
        self.params = params
    
    def process(self, image):
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return image
```

### 2.2 通道工作流程

1. **初始化**：创建通道实例时设置修复参数
2. **处理**：调用 `process()` 方法对输入图像应用修复
3. **输出**：返回修复后的图像

## 3. 已有修复通道

### 3.1 双边滤波通道

**类名**：`BilateralFilterChannel`
**功能**：使用双边滤波进行去噪，同时保留边缘
**参数**：
- `d`：滤波直径，默认 9
- `sigma_color`：颜色空间标准差，默认 75
- `sigma_space`：坐标空间标准差，默认 75

**效果**：有效去除噪声，同时保持边缘清晰

### 3.2 非局部均值去噪通道

**类名**：`NonLocalMeansChannel`
**功能**：使用非局部均值算法进行去噪
**参数**：
- `h`：滤波强度，默认 10
- `templateWindowSize`：模板窗口大小，默认 7
- `searchWindowSize`：搜索窗口大小，默认 21

**效果**：对高斯噪声有很好的去噪效果，能保留图像细节

### 3.3 图像锐化通道

**类名**：`SharpenChannel`
**功能**：增强图像边缘和细节
**参数**：
- `kernel_size`：高斯模糊核大小，默认 (3, 3)
- `sigma`：高斯标准差，默认 1.0
- `amount`：锐化强度，默认 1.0

**效果**：增强图像清晰度，改善模糊效果

### 3.4 维纳滤波通道

**类名**：`WienerFilterChannel`
**功能**：使用维纳滤波进行去模糊
**参数**：
- `kernel_size`：模糊核大小，默认 5
- `sigma`：高斯模糊核的标准差，默认 2
- `K`：噪声功率与信号功率的比值，默认 0.01

**效果**：对高斯模糊有较好的修复效果

### 3.5 Richardson-Lucy 通道

**类名**：`RichardsonLucyChannel`
**功能**：使用 Richardson-Lucy 算法进行去模糊
**参数**：
- `kernel_size`：模糊核大小，默认 5
- `sigma`：高斯模糊核的标准差，默认 2
- `iterations`：迭代次数，默认 30

**效果**：对各种模糊类型都有较好的修复效果，尤其是运动模糊

### 3.6 超分辨率通道

**类名**：`SuperResolutionChannel`
**功能**：提高图像分辨率
**参数**：
- `scale`：放大倍数，默认 2

**效果**：恢复低分辨率图像的细节

### 3.7 复合修复通道

**类名**：`CompositeRestorationChannel`
**功能**：顺序应用多个修复通道
**参数**：
- `channels`：修复通道列表

**方法**：
- `add_channel(channel)`：添加修复通道
- `process(image)`：按顺序应用所有通道
- `get_channel_names()`：获取通道名称列表

**效果**：组合多种修复方法，处理复杂的退化场景

## 4. 效果对比

| 通道类型 | 适用场景 | 优势 | 局限性 |
|---------|---------|------|--------|
| 双边滤波 | 高斯噪声、椒盐噪声 | 边缘保持好 | 对强噪声效果有限 |
| 非局部均值 | 高斯噪声 | 去噪效果好，细节保留 | 计算量大 |
| 图像锐化 | 模糊、对比度低 | 增强细节 | 可能放大噪声 |
| 维纳滤波 | 高斯模糊 | 数学上最优 | 对噪声敏感 |
| Richardson-Lucy | 各种模糊 | 效果全面 | 计算量大，可能振铃 |
| 超分辨率 | 低分辨率 | 提高细节 | 不能创造不存在的信息 |

## 5. 创建新的修复通道

### 5.1 步骤

1. **继承基类**：创建一个继承自 `BaseRestorationChannel` 的新类
2. **实现构造函数**：初始化修复参数
3. **重写 process 方法**：实现修复逻辑
4. **更新导入**：在 `channels/__init__.py` 中添加新通道

### 5.2 示例：创建新的修复通道

```python
# custom_restoration.py
import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class CustomRestorationChannel(BaseRestorationChannel):
    def __init__(self, param1=1, param2=2):
        super().__init__(param1=param1, param2=param2)
        self.param1 = param1
        self.param2 = param2
    
    def process(self, image):
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 实现自定义修复逻辑
        # ...
        
        return restored_image

# 在 channels/__init__.py 中添加
from .custom_restoration import CustomRestorationChannel

__all__ = [
    # 其他通道...
    'CustomRestorationChannel'
]
```

### 5.3 最佳实践

- **参数验证**：验证输入参数的有效性
- **边界处理**：处理图像边界情况
- **性能考虑**：优化计算密集型操作
- **文档**：为新通道添加详细的文档注释
- **测试**：为新通道编写测试用例
- **鲁棒性**：处理不同类型的输入图像

## 6. 使用示例

### 6.1 基本使用

```python
from channels import NonLocalMeansChannel, SharpenChannel

# 创建修复通道
denoise_channel = NonLocalMeansChannel(h=15)
sharpen_channel = SharpenChannel(amount=1.2)

# 应用修复
degraded_image = load_degraded_image()
restored1 = denoise_channel.process(degraded_image)  # 只去噪
restored2 = sharpen_channel.process(degraded_image)  # 只锐化
```

### 6.2 使用复合通道

```python
from channels import (
    NonLocalMeansChannel,
    SharpenChannel,
    CompositeRestorationChannel
)

# 创建复合通道
composite_channel = CompositeRestorationChannel()
composite_channel.add_channel(NonLocalMeansChannel(h=15))  # 先去噪
composite_channel.add_channel(SharpenChannel(amount=1.2))  # 再锐化

# 应用复合修复
restored = composite_channel.process(degraded_image)
```

### 6.3 与评估系统集成

```python
from core import ChannelEvaluator

# 使用字符串标签创建通道
evaluator = ChannelEvaluator()
evaluator.evaluate(
    degradation_type='gaussian_blur',
    restoration_type='richardson_lucy',
    restoration_params={'iterations': 30},
    num_tests=5
)

# 或直接使用通道对象
from core import BaseEvaluator
from channels import GaussianBlurChannel, RichardsonLucyChannel, RecognitionChannel

degradation_channel = GaussianBlurChannel(sigma=3)
restoration_channel = RichardsonLucyChannel(iterations=30)
recognition_channel = RecognitionChannel()

evaluator = BaseEvaluator()
evaluator.evaluate(degradation_channel, restoration_channel, recognition_channel, num_tests=5)
```

## 7. 通道组合策略

不同的退化类型需要不同的修复策略，以下是一些推荐的组合：

| 退化类型 | 推荐修复组合 | 理由 |
|---------|-------------|------|
| 高斯噪声 | 非局部均值 + 锐化 | 先去噪，再增强细节 |
| 椒盐噪声 | 双边滤波 + 锐化 | 去除噪声同时保持边缘，再增强细节 |
| 运动模糊 | Richardson-Lucy + 锐化 | 先去模糊，再增强边缘 |
| 高斯模糊 | Richardson-Lucy 或 维纳滤波 + 锐化 | 先去模糊，再增强细节 |
| 散焦模糊 | Richardson-Lucy + 锐化 | 恢复焦点，再增强细节 |
| 下采样 | 超分辨率 + 锐化 | 提高分辨率，再增强细节 |

## 8. 总结

修复通道系统提供了一个灵活、可扩展的框架，用于实现和评估各种图像修复算法。通过使用不同的修复通道，可以针对不同的退化类型选择合适的修复方法。同时，该系统的模块化设计使得添加新的修复算法变得简单直接，为未来的扩展提供了良好的基础。

修复通道的选择应根据具体的退化类型和修复目标来确定，有时组合使用多种修复方法可以获得更好的效果。通过评估系统，可以客观地比较不同修复方法的性能，为实际应用选择最佳的修复策略。
