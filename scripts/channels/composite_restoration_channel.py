import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class CompositeRestorationChannel(BaseRestorationChannel):
    """复合修复通道"""
    def __init__(self, channels=None):
        """初始化复合修复通道
        
        Args:
            channels: 修复通道列表
        """
        super().__init__(channels=channels)
        self.channels = channels if channels else []
    
    def add_channel(self, channel):
        """添加修复通道
        
        Args:
            channel: 修复通道实例
        """
        if isinstance(channel, BaseRestorationChannel):
            self.channels.append(channel)
        else:
            raise ValueError("通道必须是 BaseRestorationChannel 的实例")
    
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
        
        # 依次应用每个修复通道
        result = image.copy()
        for channel in self.channels:
            result = channel.process(result)
        
        return result
    
    def get_channel_names(self):
        """获取通道名称列表
        
        Returns:
            list: 通道名称列表
        """
        return [type(channel).__name__ for channel in self.channels]
