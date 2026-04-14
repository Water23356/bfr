import os
import sys
import cv2
import numpy as np

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import (
    GaussianNoiseChannel,
    MotionBlurChannel,
    GaussianBlurChannel,
    CompositeDegradationChannel
)

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def save_image(image, output_path):
    """保存图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存图像到: {output_path}")

def test_composite_degradation():
    """测试复合退化通道"""
    print("===== 测试复合退化通道 =====")
    
    # 创建一个简单的测试图像（使用随机生成的图像作为示例）
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # 保存原始图像
    save_image(test_image, '../data/test/original_image.jpg')
    
    # 创建复合退化通道
    composite_channel = CompositeDegradationChannel()
    
    # 添加退化通道
    composite_channel.add_channel(GaussianNoiseChannel(sigma=15))  # 先添加高斯噪声
    composite_channel.add_channel(MotionBlurChannel(kernel_size=10, angle=45))  # 再添加运动模糊
    composite_channel.add_channel(GaussianBlurChannel(sigma=2))  # 最后添加高斯模糊
    
    # 应用复合退化
    degraded_image = composite_channel.process(test_image)
    
    # 保存退化图像
    save_image(degraded_image, '../data/test/composite_degraded.jpg')
    
    # 打印通道信息
    print(f"\n复合通道包含的退化通道: {composite_channel.get_channel_names()}")
    print("\n测试完成！")

if __name__ == "__main__":
    test_composite_degradation()
