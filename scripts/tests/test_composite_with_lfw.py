import os
import sys
import cv2
import numpy as np
from data_provider import DataProvider
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

def save_image(image, output_path):
    """保存图像"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"保存图像到: {output_path}")

def test_composite_with_lfw():
    """使用 LFW 数据集测试复合修复通道"""
    print("===== 使用 LFW 数据集测试复合修复通道 =====")
    
    # 加载 LFW 数据集
    data_provider = DataProvider()
    data = data_provider.load_lfw_dataset()
    if data is None:
        print("无法加载数据集，测试终止")
        return
    
    # 获取一个人的图像
    person_index = data_provider.get_random_person(min_faces=1)
    if person_index is None:
        print("无法找到合适的人物，测试终止")
        return
    
    person_images = data_provider.get_person_images(person_index, num_images=1)
    if len(person_images) == 0:
        print("无法获取人物图像，测试终止")
        return
    
    original_image = person_images[0]
    
    # 保存原始图像
    save_image(original_image, '../data/test/original_image.jpg')
    
    # 1. 测试：高斯噪声 + 非局部均值去噪 + 锐化
    print("\n1. 测试：高斯噪声 + 非局部均值去噪 + 锐化")
    degradation_channel = DegradationChannel('gaussian_noise', sigma=25)
    degraded_image = degradation_channel.process(original_image)
    save_image(degraded_image, '../data/test/degraded_noise.jpg')
    
    composite_channel1 = CompositeRestorationChannel()
    composite_channel1.add_channel(NonLocalMeansChannel(h=15))
    composite_channel1.add_channel(SharpenChannel(amount=1.2))
    
    restored_image1 = composite_channel1.process(degraded_image)
    save_image(restored_image1, '../data/test/restored_noise.jpg')
    print(f"  通道顺序: {composite_channel1.get_channel_names()}")
    
    # 2. 测试：高斯模糊 + Richardson-Lucy + 锐化
    print("\n2. 测试：高斯模糊 + Richardson-Lucy + 锐化")
    degradation_channel = DegradationChannel('gaussian_blur', sigma=3)
    degraded_image = degradation_channel.process(original_image)
    save_image(degraded_image, '../data/test/degraded_blur.jpg')
    
    composite_channel2 = CompositeRestorationChannel()
    composite_channel2.add_channel(RichardsonLucyChannel(iterations=30))
    composite_channel2.add_channel(SharpenChannel(amount=1.0))
    
    restored_image2 = composite_channel2.process(degraded_image)
    save_image(restored_image2, '../data/test/restored_blur.jpg')
    print(f"  通道顺序: {composite_channel2.get_channel_names()}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_composite_with_lfw()
