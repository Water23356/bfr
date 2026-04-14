# 项目 README

## 项目描述

这是一个 Python 项目。

## 环境依赖

### Python 版本
- Python 3.7+

### 依赖包

使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 项目结构

- `data/` - 数据目录（被忽略）
- `logs/` - 日志目录（被忽略）
- `temp/` - 临时文件目录（被忽略）
- `doc/` - 技术文档目录
- `scripts/` - 项目脚本目录
- `models/` - 模型目录（被忽略）

## 技术文档

相关技术文档位于 `doc/` 目录下，包含以下文件：
- `degradation_channels_doc.md` - 退化通道文档
- `restoration_channels_doc.md` - 修复通道文档
- `environment_dependencies.md` - 环境依赖文档
- `图像修复指标.md` - 图像修复评估指标文档

## 生成退化数据集

### 功能说明

项目提供了生成退化数据集的脚本，可以将 LFW 人脸数据集转换为各种退化类型的数据集，用于评估图像修复算法的性能。

### 核心函数说明

`generate` 函数是生成退化数据集的核心方法，其签名如下：

```python
def generate(self, degradation_type=None, num_images=100, degradation_channel=None, **params):
```

**参数说明：**
- `degradation_type`：退化类型（当 `degradation_channel` 为 None 时使用）
- `num_images`：生成的图像数量
- `degradation_channel`：外部传入的退化通道对象，**优先使用**
- `**params`：退化参数

**重要说明：** 该函数优先使用 `degradation_channel` 参数生成退化数据集。如果提供了 `degradation_channel`，则会忽略 `degradation_type` 参数，并使用传入的退化通道对象进行处理。

### 退化通道介绍

退化通道是实现图像退化效果的核心组件，每个退化通道对应一种特定的退化类型。项目提供了多种退化通道，包括：

- **运动模糊通道**：模拟相机或物体运动导致的模糊
- **高斯模糊通道**：模拟失焦或大气散射导致的模糊
- **散焦模糊通道**：模拟镜头散焦导致的模糊
- **随机组合通道**：随机组合多种模糊效果
- **噪声通道**：添加高斯噪声或椒盐噪声

**详细参考文档**：`doc/degradation_channels_doc.md`

### 使用样例

#### 命令行方式

```bash
# 生成运动模糊退化数据集
python scripts/cli/generate_degraded_dataset.py --degradation motion_blur --num-images 100

# 生成高斯模糊退化数据集
python scripts/cli/generate_degraded_dataset.py --degradation gaussian_blur --num-images 100 --sigma 3

# 生成散焦模糊退化数据集
python scripts/cli/generate_degraded_dataset.py --degradation defocus_blur --num-images 100 --radius 5

# 生成随机组合模糊退化数据集
python scripts/cli/generate_degraded_dataset.py --degradation random_blur --num-images 100

# 生成所有模糊类型的退化数据集
python scripts/cli/generate_degraded_dataset.py --all --num-images 50

# 指定输出目录
python scripts/cli/generate_degraded_dataset.py --degradation motion_blur --num-images 100 --output-dir ./custom_output
```

#### Python 代码方式

**方式 1：使用退化类型字符串**

```python
from scripts.cli.generate_degraded_dataset import DegradedDatasetGenerator

# 创建生成器
generator = DegradedDatasetGenerator(output_dir='./data/degraded')

# 生成运动模糊数据集
generator.generate('motion_blur', num_images=100, kernel_size=15, angle=45, sigma=2)

# 生成高斯模糊数据集
generator.generate('gaussian_blur', num_images=100, sigma=3)

# 生成所有模糊类型的数据集
generator.generate_all_blur_types(num_images=50)
```

**方式 2：使用退化通道对象（优先推荐）**

```python
from scripts.cli.generate_degraded_dataset import DegradedDatasetGenerator
from scripts.channels import GaussianNoiseChannel, SaltAndPepperChannel

# 创建生成器
generator = DegradedDatasetGenerator(output_dir='./data/degraded')

# 创建高斯噪声退化通道
noise_channel = GaussianNoiseChannel(mean=0, sigma=25)
# 使用退化通道对象生成数据集
generator.generate(degradation_channel=noise_channel, num_images=100)

# 创建椒盐噪声退化通道
salt_pepper_channel = SaltAndPepperChannel(amount=0.05)
# 使用退化通道对象生成数据集
generator.generate(degradation_channel=salt_pepper_channel, num_images=100)
```

### 支持的退化类型

- `motion_blur` - 运动模糊
- `gaussian_blur` - 高斯模糊
- `defocus_blur` - 散焦模糊
- `random_blur` - 随机组合模糊
- `gaussian_noise` - 高斯噪声
- `salt_and_pepper` - 椒盐噪声
- `downsample` - 降采样
