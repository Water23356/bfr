import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class DefocusBlurChannel(BaseDegradationChannel):
    """散焦模糊退化通道"""
    def __init__(self, radius=5):
        """初始化散焦模糊通道
        
        Args:
            radius: 圆盘半径（2~8）
        """
        # 确保半径在有效范围内
        radius = max(2, min(8, radius))
        super().__init__(radius=radius)
        self.radius = radius
    
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
        
        # 创建圆盘模糊核
        kernel_size = self.radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 生成圆盘
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = np.sqrt((i - self.radius)**2 + (j - self.radius)**2)
                if distance <= self.radius:
                    kernel[i, j] = 1
        
        # 归一化模糊核
        kernel = kernel / kernel.sum()
        
        # 应用模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image
