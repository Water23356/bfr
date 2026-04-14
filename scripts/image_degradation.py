import cv2
import numpy as np
import random

class ImageDegradation:
    def __init__(self):
        """初始化图像退化模块"""
        pass
    
    def add_gaussian_noise(self, image, mean=0, sigma=25):
        """添加高斯噪声
        
        Args:
            image: 输入图像 (0-255)
            mean: 噪声均值
            sigma: 噪声标准差
            
        Returns:
            numpy.ndarray: 带噪声的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 添加高斯噪声
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        
        return noisy_image
    
    def add_salt_and_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """添加椒盐噪声
        
        Args:
            image: 输入图像 (0-255)
            salt_prob: 盐噪声概率
            pepper_prob: 椒噪声概率
            
        Returns:
            numpy.ndarray: 带噪声的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        noisy_image = np.copy(image)
        total_pixels = image.size
        
        # 添加盐噪声（白色像素）
        num_salt = int(total_pixels * salt_prob)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 255
        
        # 添加椒噪声（黑色像素）
        num_pepper = int(total_pixels * pepper_prob)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[coords[0], coords[1]] = 0
        
        return noisy_image
    
    def motion_blur(self, image, kernel_size=15, angle=45):
        """添加运动模糊
        
        Args:
            image: 输入图像
            kernel_size: 模糊核大小
            angle: 模糊方向（角度）
            
        Returns:
            numpy.ndarray: 模糊后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 创建运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        angle_rad = np.deg2rad(angle)
        
        # 计算模糊核的中心点
        center = kernel_size // 2
        
        # 计算模糊方向上的像素
        for i in range(kernel_size):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        
        # 归一化模糊核
        kernel = kernel / kernel.sum()
        
        # 应用模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        
        return blurred_image
    
    def gaussian_blur(self, image, kernel_size=(5, 5), sigma=0):
        """添加高斯模糊
        
        Args:
            image: 输入图像
            kernel_size: 模糊核大小
            sigma: 高斯标准差
            
        Returns:
            numpy.ndarray: 模糊后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image
    
    def downsample(self, image, scale=0.5):
        """下采样（降低分辨率）
        
        Args:
            image: 输入图像
            scale: 缩放比例
            
        Returns:
            numpy.ndarray: 下采样后的图像
        """
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 计算新尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # 下采样
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 恢复到原始尺寸（模拟低分辨率效果）
        restored = cv2.resize(downsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return restored
    
    def apply_degradation(self, image, degradation_type, **kwargs):
        """应用指定类型的退化
        
        Args:
            image: 输入图像
            degradation_type: 退化类型
            **kwargs: 退化参数
            
        Returns:
            numpy.ndarray: 退化后的图像
        """
        degradation_functions = {
            'gaussian_noise': self.add_gaussian_noise,
            'salt_and_pepper': self.add_salt_and_pepper_noise,
            'motion_blur': self.motion_blur,
            'gaussian_blur': self.gaussian_blur,
            'downsample': self.downsample
        }
        
        if degradation_type not in degradation_functions:
            raise ValueError(f"不支持的退化类型: {degradation_type}")
        
        return degradation_functions[degradation_type](image, **kwargs)
