import cv2
import numpy as np

class BaseDegradationChannel:
    """退化通道基类"""
    def __init__(self, **params):
        """初始化退化通道
        
        Args:
            **params: 退化参数
        """
        self.params = params
    
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
        return image
