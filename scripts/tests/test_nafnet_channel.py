import os
import sys
import cv2
import numpy as np

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import (
    GaussianNoiseChannel,
    NAFNetChannel
)

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def save_image(image, output_path):
    """保存图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存图像到: {output_path}")

def test_nafnet_channel():
    """测试 NAFNet 通道"""
    print("===== 测试 NAFNet 通道 =====")
    
    # 创建一个简单的测试图像（使用随机生成的图像作为示例）
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 保存原始图像
    save_image(test_image, '../data/test/original_nafnet.jpg')
    
    # 添加高斯噪声
    noise_channel = GaussianNoiseChannel(sigma=25)
    degraded_image = noise_channel.process(test_image)
    save_image(degraded_image, '../data/test/degraded_nafnet.jpg')
    
    # 创建 NAFNet 通道
    try:
        nafnet_channel = NAFNetChannel()
        print("NAFNet 通道初始化成功")
        
        # 应用 NAFNet 修复
        restored_image = nafnet_channel.process(degraded_image)
        save_image(restored_image, '../data/test/restored_nafnet.jpg')
        print("NAFNet 修复完成")
        
    except Exception as e:
        print(f"NAFNet 通道测试失败: {e}")
        print("注意：NAFNet 模型需要预训练权重才能获得最佳效果")

if __name__ == "__main__":
    test_nafnet_channel()
