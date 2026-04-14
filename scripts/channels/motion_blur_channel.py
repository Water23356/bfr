import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class MotionBlurChannel(BaseDegradationChannel):
    """运动模糊退化通道"""
    def __init__(self, kernel_size=15, angle=45, sigma=2):
        """初始化运动模糊通道
        
        Args:
            kernel_size: 模糊核大小
            angle: 模糊方向（角度）
            sigma: 高斯运动核的标准差
        """
        super().__init__(kernel_size=kernel_size, angle=angle, sigma=sigma)
        self.kernel_size = kernel_size
        self.angle = angle
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
        
        # 创建高斯运动模糊核
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        angle_rad = np.deg2rad(self.angle)
        
        # 计算模糊核的中心点
        center = self.kernel_size // 2
        
        # 计算模糊方向上的像素并应用高斯权重
        for i in range(self.kernel_size):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                # 计算高斯权重
                distance = np.sqrt((x - center)**2 + (y - center)**2)
                weight = np.exp(-(distance**2) / (2 * self.sigma**2))
                kernel[y, x] = weight
        
        # 归一化模糊核
        kernel = kernel / kernel.sum()
        
        # 应用模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image
