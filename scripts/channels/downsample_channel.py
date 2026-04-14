import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class DownsampleChannel(BaseDegradationChannel):
    """下采样退化通道"""
    def __init__(self, scale=0.5):
        """初始化下采样通道
        
        Args:
            scale: 缩放比例
        """
        super().__init__(scale=scale)
        self.scale = scale
    
    def process(self, image):
        """处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            numpy.ndarray: 退化后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 计算新尺寸
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)
        
        # 下采样
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 恢复到原始尺寸（模拟低分辨率效果）
        restored = cv2.resize(downsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return restored
