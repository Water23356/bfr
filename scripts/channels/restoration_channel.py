import cv2
import numpy as np

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
        elif self.restoration_type == 'richardson_lucy':
            return self._deblur_richardson_lucy(image, **self.params)
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
    
    def _deblur_wiener(self, image, kernel_size=5, sigma=2, K=0.01):
        """维纳滤波去模糊
        
        Args:
            image: 输入图像
            kernel_size: 模糊核大小
            sigma: 高斯模糊核的标准差
            K: 噪声功率与信号功率的比值
            
        Returns:
            numpy.ndarray: 去模糊后的图像
        """
        # 转换为灰度图像（如果是彩色图像）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        
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
    
    def _deblur_richardson_lucy(self, image, kernel_size=5, sigma=2, iterations=30):
        """Richardson-Lucy 算法去模糊
        
        Args:
            image: 输入图像
            kernel_size: 模糊核大小
            sigma: 高斯模糊核的标准差
            iterations: 迭代次数
            
        Returns:
            numpy.ndarray: 去模糊后的图像
        """
        # 转换为灰度图像（如果是彩色图像）
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 转换为浮点型
        image = image.astype(np.float32)
        
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        kernel = kernel / kernel.sum()
        
        # 初始化估计图像
        estimate = np.copy(image)
        
        # 执行 Richardson-Lucy 迭代
        for i in range(iterations):
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
    
    def _super_resolution(self, image, scale=2):
        """简单超分辨率（双线性插值）"""
        # 计算新尺寸
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        
        # 应用双线性插值
        restored = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return restored
