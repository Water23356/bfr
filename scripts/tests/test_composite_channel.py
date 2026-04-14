import os
import sys
import cv2
import numpy as np
from channels import (
    DegradationChannel,
    BilateralFilterChannel,
    NonLocalMeansChannel,
    SharpenChannel,
    RichardsonLucyChannel,
    CompositeRestorationChannel
)

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def load_image(image_path):
    """加载图像"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return image

def save_image(image, output_path):
    """保存图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存图像到: {output_path}")

def test_composite_channel():
    """测试复合修复通道"""
    print("===== 测试复合修复通道 =====")
    
    # 创建一个简单的测试图像（使用 Lena 图像）
    # 这里我们使用随机生成的图像作为示例
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # 应用高斯噪声
    degradation_channel = DegradationChannel('gaussian_noise', sigma=25)
    degraded_image = degradation_channel.process(test_image)
    
    # 保存退化图像
    save_image(degraded_image, '../data/test/degraded_image.jpg')
    
    # 创建复合修复通道
    composite_channel = CompositeRestorationChannel()
    
    # 添加修复通道
    composite_channel.add_channel(NonLocalMeansChannel(h=15))  # 去噪
    composite_channel.add_channel(SharpenChannel(amount=1.5))    # 锐化
    
    # 应用复合修复
    restored_image = composite_channel.process(degraded_image)
    
    # 保存修复图像
    save_image(restored_image, '../data/test/restored_image.jpg')
    
    # 打印通道信息
    print(f"\n复合通道包含的修复通道: {composite_channel.get_channel_names()}")
    print("\n测试完成！")

if __name__ == "__main__":
    test_composite_channel()
