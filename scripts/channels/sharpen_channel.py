import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class SharpenChannel(BaseRestorationChannel):
    """图像锐化修复通道"""
    def __init__(self, kernel_size=(3, 3), sigma=1.0, amount=1.0):
        """初始化图像锐化通道
        
        Args:
            kernel_size: 高斯模糊核大小
            sigma: 高斯标准差
            amount: 锐化强度
        """
        super().__init__(kernel_size=kernel_size, sigma=sigma, amount=amount)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
    
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
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        
        # 计算锐化图像
        sharpened = cv2.addWeighted(image, 1 + self.amount, blurred, -self.amount, 0)
        return sharpened
