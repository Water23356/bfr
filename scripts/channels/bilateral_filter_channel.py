import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class BilateralFilterChannel(BaseRestorationChannel):
    """双边滤波修复通道"""
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        """初始化双边滤波通道
        
        Args:
            d: 滤波直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
        """
        super().__init__(d=d, sigma_color=sigma_color, sigma_space=sigma_space)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
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
        
        # 应用双边滤波
        restored = cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)
        return restored
