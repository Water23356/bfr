import os
import sys
import cv2
import numpy as np

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels import NAFNetChannel, GaussianBlurChannel

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def save_image(image, output_path):
    """保存图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存图像到: {output_path}")

def test_external_model_path():
    """测试从外部传入模型文件路径"""
    print("===== 测试从外部传入模型文件路径 =====")
    
    # 创建一个简单的测试图像
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 保存原始图像
    save_image(test_image, '../data/test/original_external.jpg')
    
    # 添加高斯噪声
    blur_channel = GaussianBlurChannel(sigma=3)
    degraded_image = blur_channel.process(test_image)
    save_image(degraded_image, '../data/test/degraded_external.jpg')
    
    # 测试 1: 不指定模型路径（使用默认路径）
    print("\n测试 1: 不指定模型路径")
    nafnet_channel1 = NAFNetChannel()
    restored1 = nafnet_channel1.process(degraded_image)
    save_image(restored1, '../data/test/restored_default.jpg')
    
    # 测试 2: 指定不存在的模型路径
    print("\n测试 2: 指定不存在的模型路径")
    non_existent_path = '../models/non_existent_model.pth'
    nafnet_channel2 = NAFNetChannel(model_path=non_existent_path)
    restored2 = nafnet_channel2.process(degraded_image)
    save_image(restored2, '../data/test/restored_nonexistent.jpg')
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_external_model_path()
