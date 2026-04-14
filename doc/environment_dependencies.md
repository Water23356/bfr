# 项目环境依赖

本文档记录了项目所需的 Python 包和版本，以确保项目能够正常运行。

## 基础环境

- **Python**：3.10 或更高版本
- **Anaconda**：用于环境管理

## 主要依赖包

### 核心依赖

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| numpy | >= 1.20.0 | 数值计算 | `pip install numpy` |
| opencv-python | >= 4.5.0 | 图像处理 | `pip install opencv-python` |
| deepface | >= 0.0.90 | 人脸识别 | `pip install deepface` |
| scikit-learn | >= 1.0.0 | 机器学习工具 | `pip install scikit-learn` |
| scikit-image | >= 0.18.0 | 图像处理 | `pip install scikit-image` |
| matplotlib | >= 3.5.0 | 数据可视化 | `pip install matplotlib` |

### 可选依赖（用于 NAFNet 模型）

| 包名 | 版本 | 用途 | 安装命令 |
|------|------|------|----------|
| torch | >= 2.0.0 | PyTorch 框架 | `pip install torch torchvision` |
| torchvision | >= 0.15.0 | PyTorch 视觉工具 | 包含在上述命令中 |

## 环境配置步骤

### 1. 创建 Anaconda 环境

```bash
conda create -n facereco python=3.10
conda activate facereco
```

### 2. 安装核心依赖

```bash
pip install numpy opencv-python deepface scikit-learn scikit-image matplotlib
```

### 3. 安装可选依赖（用于 NAFNet）

```bash
pip install torch torchvision
```

## 验证安装

运行以下命令验证所有依赖是否正确安装：

```bash
python -c "import numpy; import cv2; import deepface; import sklearn; import skimage; import matplotlib; print('All core dependencies are installed successfully!')"

# 验证 PyTorch（如果安装了）
python -c "import torch; import torchvision; print('PyTorch is installed successfully!')"
```

## 常见问题

### 1. DeepFace 模型下载失败

**问题**：运行时出现模型下载失败的错误。

**解决方案**：
- 确保网络连接正常
- 手动下载模型文件到 `models/.deepface/weights/` 目录
- 模型下载地址：https://github.com/serengil/deepface_models/releases

### 2. PyTorch 安装失败

**问题**：PyTorch 安装过程中出现错误。

**解决方案**：
- 确保 Python 版本为 3.10 或更高
- 尝试使用官方推荐的安装命令：https://pytorch.org/get-started/locally/
- 对于没有 GPU 的系统，使用 CPU 版本：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

### 3. 内存不足

**问题**：运行时出现内存不足的错误。

**解决方案**：
- 减少测试图像的数量
- 减小图像的尺寸
- 关闭其他占用内存的程序

## 依赖包说明

### numpy
- **用途**：提供高性能的数值计算功能，用于图像处理和矩阵运算
- **版本要求**：1.20.0 或更高，以确保与其他依赖包的兼容性

### opencv-python
- **用途**：提供图像处理功能，包括图像读取、转换、滤波等
- **版本要求**：4.5.0 或更高，以支持最新的图像处理功能

### deepface
- **用途**：提供人脸识别和属性分析功能
- **版本要求**：0.0.90 或更高，以确保模型下载和识别功能正常

### scikit-learn
- **用途**：提供机器学习工具，用于数据集加载和处理
- **版本要求**：1.0.0 或更高，以支持 LFW 数据集的加载

### scikit-image
- **用途**：提供高级图像处理功能
- **版本要求**：0.18.0 或更高，以支持各种图像变换和增强

### matplotlib
- **用途**：提供数据可视化功能，用于显示图像和结果
- **版本要求**：3.5.0 或更高，以支持最新的可视化功能

### torch 和 torchvision
- **用途**：提供深度学习框架，用于 NAFNet 模型的运行
- **版本要求**：2.0.0 或更高，以支持最新的深度学习功能
- **注意**：这是可选依赖，只在使用 NAFNet 模型时需要

## 系统要求

- **操作系统**：Windows、macOS 或 Linux
- **内存**：至少 4GB RAM（推荐 8GB 或更高）
- **存储空间**：至少 2GB 可用空间（用于安装依赖和模型文件）
- **CPU**：支持 SSE3、SSE4.1、SSE4.2、AVX 指令集的处理器
- **GPU**（可选）：支持 CUDA 的 NVIDIA GPU，用于加速 PyTorch 计算

## 总结

本项目的依赖包设计考虑了功能完整性和性能要求，确保在各种环境下都能正常运行。核心依赖包提供了基本的图像处理和人脸识别功能，而可选的 PyTorch 依赖则为 NAFNet 模型提供了支持。

通过按照上述步骤配置环境，用户可以确保项目能够顺利运行，并且能够充分利用所有功能。
