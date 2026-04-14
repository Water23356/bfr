import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class GaussianNoiseChannel(BaseDegradationChannel):
    """高斯噪声退化通道"""
    def __init__(self, mean=0, sigma=25):
        """初始化高斯噪声通道
        
        Args:
            mean: 噪声均值
            sigma: 噪声标准差
        """
        super().__init__(mean=mean, sigma=sigma)
        self.mean = mean
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
        
        # 添加高斯噪声
        noise = np.random.normal(self.mean, self.sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
