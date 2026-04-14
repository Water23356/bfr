import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class SaltAndPepperChannel(BaseDegradationChannel):
    """椒盐噪声退化通道"""
    def __init__(self, salt_prob=0.02, pepper_prob=0.02):
        """初始化椒盐噪声通道
        
        Args:
            salt_prob: 盐噪声概率
            pepper_prob: 椒噪声概率
        """
        super().__init__(salt_prob=salt_prob, pepper_prob=pepper_prob)
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
    
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
        
        # 添加椒盐噪声
        noisy_image = np.copy(image)
        total_pixels = image.size
        
        # 添加盐噪声（白色像素）
        num_salt = int(total_pixels * self.salt_prob)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255
        
        # 添加椒噪声（黑色像素）
        num_pepper = int(total_pixels * self.pepper_prob)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0
        
        return noisy_image
