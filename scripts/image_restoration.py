import cv2
import numpy as np
from scipy import signal

class ImageRestoration:
    def __init__(self):
        """初始化图像修复模块"""
        pass
    
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """双边滤波去噪
        
        Args:
            image: 输入图像
            d: 滤波直径
            sigma_color: 颜色空间标准差
            sigma_space: 坐标空间标准差
            
        Returns:
            numpy.ndarray: 去噪后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 应用双边滤波
        restored = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return restored
    
    def denoise_non_local_means(self, image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
        """非局部均值去噪
        
        Args:
            image: 输入图像
            h: 滤波强度
            hColor: 颜色空间滤波强度
            templateWindowSize: 模板窗口大小
            searchWindowSize: 搜索窗口大小
            
        Returns:
            numpy.ndarray: 去噪后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 应用非局部均值去噪
        restored = cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
        return restored
    
    def sharpen(self, image, kernel_size=(3, 3), sigma=1.0, amount=1.0):
        """图像锐化
        
        Args:
            image: 输入图像
            kernel_size: 高斯模糊核大小
            sigma: 高斯模糊标准差
            amount: 锐化强度
            
        Returns:
            numpy.ndarray: 锐化后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # 计算锐化图像
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def deblur_wiener(self, image, kernel, K=0.01):
        """维纳滤波去模糊
        
        Args:
            image: 输入图像
            kernel: 模糊核
            K: 噪声功率与信号功率的比值
            
        Returns:
            numpy.ndarray: 去模糊后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 转换为灰度图像（如果是彩色图像）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算傅里叶变换
        image_fft = np.fft.fft2(image)
        kernel_fft = np.fft.fft2(kernel, s=image.shape)
        
        # 计算维纳滤波
        kernel_fft_conj = np.conj(kernel_fft)
        Wiener_filter = kernel_fft_conj / (np.abs(kernel_fft)**2 + K)
        restored_fft = image_fft * Wiener_filter
        
        # 逆傅里叶变换
        restored = np.abs(np.fft.ifft2(restored_fft))
        restored = np.uint8(restored)
        
        return restored
    
    def super_resolution(self, image, scale=2):
        """简单超分辨率（双线性插值）
        
        Args:
            image: 输入图像
            scale: 放大倍数
            
        Returns:
            numpy.ndarray: 超分辨率后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 计算新尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # 应用双线性插值
        restored = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        return restored
    
    def apply_restoration(self, image, restoration_type, **kwargs):
        """应用指定类型的修复
        
        Args:
            image: 输入图像
            restoration_type: 修复类型
            **kwargs: 修复参数
            
        Returns:
            numpy.ndarray: 修复后的图像
        """
        restoration_functions = {
            'bilateral': self.denoise_bilateral,
            'non_local_means': self.denoise_non_local_means,
            'sharpen': self.sharpen,
            'wiener': self.deblur_wiener,
            'super_resolution': self.super_resolution
        }
        
        if restoration_type not in restoration_functions:
            raise ValueError(f"不支持的修复类型: {restoration_type}")
        
        return restoration_functions[restoration_type](image, **kwargs)
