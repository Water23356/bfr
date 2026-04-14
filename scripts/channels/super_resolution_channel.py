import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class SuperResolutionChannel(BaseRestorationChannel):
    """超分辨率修复通道"""
    def __init__(self, scale=2):
        """初始化超分辨率通道
        
        Args:
            scale: 放大倍数
        """
        super().__init__(scale=scale)
        self.scale = scale
    
    def process(self, image):
        """处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            numpy.ndarray: 修复后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 计算新尺寸
        new_width = int(image.shape[1] * self.scale)
        new_height = int(image.shape[0] * self.scale)
        
        # 应用双线性插值
        restored = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return restored
