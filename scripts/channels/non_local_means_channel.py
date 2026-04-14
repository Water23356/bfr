import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class NonLocalMeansChannel(BaseRestorationChannel):
    """非局部均值去噪修复通道"""
    def __init__(self, h=10, templateWindowSize=7, searchWindowSize=21):
        """初始化非局部均值去噪通道
        
        Args:
            h: 滤波强度
            templateWindowSize: 模板窗口大小
            searchWindowSize: 搜索窗口大小
        """
        super().__init__(h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize
    
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
        
        # 应用非局部均值去噪
        restored = cv2.fastNlMeansDenoising(image, None, self.h, self.templateWindowSize, self.searchWindowSize)
        return restored
