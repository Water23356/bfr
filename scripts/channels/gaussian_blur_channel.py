import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class GaussianBlurChannel(BaseDegradationChannel):
    """高斯模糊退化通道"""
    def __init__(self, kernel_size=(5, 5), sigma=0):
        """初始化高斯模糊通道
        
        Args:
            kernel_size: 模糊核大小
            sigma: 高斯标准差
        """
        super().__init__(kernel_size=kernel_size, sigma=sigma)
        self.kernel_size = kernel_size
        self.sigma = sigma
    
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
        
        # 应用高斯模糊
        blurred_image = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        return blurred_image
