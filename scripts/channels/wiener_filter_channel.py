import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class WienerFilterChannel(BaseRestorationChannel):
    """维纳滤波修复通道"""
    def __init__(self, kernel_size=5, sigma=2, K=0.01):
        """初始化维纳滤波通道
        
        Args:
            kernel_size: 模糊核大小
            sigma: 高斯模糊核的标准差
            K: 噪声功率与信号功率的比值
        """
        super().__init__(kernel_size=kernel_size, sigma=sigma, K=K)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.K = K
    
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
        
        # 转换为灰度图像（如果是彩色图像）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(self.kernel_size, self.sigma)
        kernel = kernel * kernel.T
        
        # 计算傅里叶变换
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        
        # 计算维纳滤波
        kernel_fft_conj = np.conj(kernel_fft)
        Wiener_filter = kernel_fft_conj / (np.abs(kernel_fft)**2 + self.K)
        restored_fft = image_fft * Wiener_filter
        
        # 逆傅里叶变换
        restored = np.abs(np.fft.ifft2(restored_fft))
        restored = np.uint8(restored)
        return restored
