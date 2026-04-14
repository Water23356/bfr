import cv2
import numpy as np
from .base_degradation_channel import BaseDegradationChannel

class CompositeDegradationChannel(BaseDegradationChannel):
    """复合退化通道"""
    def __init__(self, channels=None):
        """初始化复合退化通道
        
        Args:
            channels: 退化通道列表
        """
        super().__init__(channels=channels)
        self.channels = channels if channels else []
    
    def add_channel(self, channel):
        """添加退化通道
        
        Args:
            channel: 退化通道实例
        """
        if isinstance(channel, BaseDegradationChannel):
            self.channels.append(channel)
        else:
            raise ValueError("通道必须是 BaseDegradationChannel 的实例")
    
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
        
        # 依次应用每个退化通道
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
