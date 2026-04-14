import os
import cv2
import numpy as np
from deepface import DeepFace

class DegradationChannel:
    """退化处理通道"""
    def __init__(self, degradation_type, **params):
        """初始化退化通道
        
        Args:
            degradation_type: 退化类型
            **params: 退化参数
        """
        self.degradation_type = degradation_type
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
        
        if self.degradation_type == 'gaussian_noise':
            return self._add_gaussian_noise(image, **self.params)
        elif self.degradation_type == 'salt_and_pepper':
            return self._add_salt_and_pepper_noise(image, **self.params)
        elif self.degradation_type == 'motion_blur':
            return self._motion_blur(image, **self.params)
        elif self.degradation_type == 'gaussian_blur':
            return self._gaussian_blur(image, **self.params)
        elif self.degradation_type == 'downsample':
            return self._downsample(image, **self.params)
        else:
            raise ValueError(f"不支持的退化类型: {self.degradation_type}")
    
    def _add_gaussian_noise(self, image, mean=0, sigma=25):
        """添加高斯噪声"""
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image
    
    def _add_salt_and_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):
        """添加椒盐噪声"""
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
    
    def _motion_blur(self, image, kernel_size=15, angle=45):
        """添加运动模糊"""
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
    
    def _gaussian_blur(self, image, kernel_size=(5, 5), sigma=0):
        """添加高斯模糊"""
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image
    
    def _downsample(self, image, scale=0.5):
        """下采样（降低分辨率）"""
        # 计算新尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # 下采样
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 恢复到原始尺寸（模拟低分辨率效果）
        restored = cv2.resize(downsampled, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return restored

class RestorationChannel:
    """图像修复通道"""
    def __init__(self, restoration_type, **params):
        """初始化修复通道
        
        Args:
            restoration_type: 修复类型
            **params: 修复参数
        """
        self.restoration_type = restoration_type
        self.params = params
    
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
        
        if self.restoration_type == 'bilateral':
            return self._denoise_bilateral(image, **self.params)
        elif self.restoration_type == 'non_local_means':
            return self._denoise_non_local_means(image, **self.params)
        elif self.restoration_type == 'sharpen':
            return self._sharpen(image, **self.params)
        elif self.restoration_type == 'wiener':
            return self._deblur_wiener(image, **self.params)
        elif self.restoration_type == 'super_resolution':
            return self._super_resolution(image, **self.params)
        else:
            raise ValueError(f"不支持的修复类型: {self.restoration_type}")
    
    def _denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """双边滤波去噪"""
        restored = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return restored
    
    def _denoise_non_local_means(self, image, h=10, templateWindowSize=7, searchWindowSize=21):
        """非局部均值去噪"""
        restored = cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
        return restored
    
    def _sharpen(self, image, kernel_size=(3, 3), sigma=1.0, amount=1.0):
        """图像锐化"""
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        
        # 计算锐化图像
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened
    
    def _deblur_wiener(self, image, kernel, K=0.01):
        """维纳滤波去模糊"""
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
    
    def _super_resolution(self, image, scale=2):
        """简单超分辨率（双线性插值）"""
        # 计算新尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # 应用双线性插值
        restored = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return restored

class RecognitionChannel:
    """人脸识别通道"""
    def __init__(self, model_name='VGG-Face', model_dir='../models'):
        """初始化识别通道
        
        Args:
            model_name: 人脸识别模型名称
            model_dir: 模型存储目录
        """
        # 设置 DEEPFACE_HOME 环境变量
        os.environ['DEEPFACE_HOME'] = model_dir
        self.model_name = model_name
        self.model_dir = model_dir
    
    def verify(self, image1, image2):
        """验证两张图像是否属于同一个人
        
        Args:
            image1: 第一张图像
            image2: 第二张图像
            
        Returns:
            bool: 是否是同一个人
            float: 相似度距离
        """
        try:
            # 保存临时图像
            img1_path = self._save_temp_image(image1, 'img1.jpg')
            img2_path = self._save_temp_image(image2, 'img2.jpg')
            
            # 使用 DeepFace 验证
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                enforce_detection=True
            )
            
            return result['verified'], result['distance']
        except Exception as e:
            print(f"识别失败: {e}")
            return False, float('inf')
    
    def _save_temp_image(self, image, filename):
        """保存临时图像"""
        temp_dir = '../temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 保存图像
        image_path = os.path.join(temp_dir, filename)
        cv2.imwrite(image_path, image)
        return image_path
