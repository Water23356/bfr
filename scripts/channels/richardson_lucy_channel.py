import cv2
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

class RichardsonLucyChannel(BaseRestorationChannel):
    """Richardson-Lucy 修复通道"""
    def __init__(self, kernel_size=5, sigma=2, iterations=30):
        """初始化 Richardson-Lucy 通道
        
        Args:
            kernel_size: 模糊核大小
            sigma: 高斯模糊核的标准差
            iterations: 迭代次数
        """
        super().__init__(kernel_size=kernel_size, sigma=sigma, iterations=iterations)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.iterations = iterations
    
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
        
        # 转换为浮点型
        image = image.astype(np.float32)
        
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(self.kernel_size, self.sigma)
        kernel = kernel * kernel.T
        kernel = kernel / kernel.sum()
        
        # 初始化估计图像
        estimate = np.copy(image)
        
        # 执行 Richardson-Lucy 迭代
        for i in range(self.iterations):
            # 前向模糊
            blurred = cv2.filter2D(estimate, -1, kernel)
            
            # 计算比值
            ratio = image / (blurred + 1e-10)  # 避免除零
            
            # 反向模糊
            ratio_blurred = cv2.filter2D(ratio, -1, np.flip(kernel))
            
            # 更新估计
            estimate *= ratio_blurred
        
        # 归一化并转换回 uint8
        estimate = np.clip(estimate, 0, 255)
        estimate = estimate.astype(np.uint8)
        return estimate
