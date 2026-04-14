import cv2
import numpy as np

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
        elif self.degradation_type == 'defocus_blur':
            return self._defocus_blur(image, **self.params)
        elif self.degradation_type == 'random_blur':
            return self._random_blur_combination(image, **self.params)
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
    
    def _motion_blur(self, image, kernel_size=15, angle=45, sigma=2):
        """添加运动模糊
        
        Args:
            image: 输入图像
            kernel_size: 模糊核大小
            angle: 模糊方向（角度）
            sigma: 高斯运动核的标准差
            
        Returns:
            numpy.ndarray: 模糊后的图像
        """
        # 创建高斯运动模糊核
        kernel = np.zeros((kernel_size, kernel_size))
        angle_rad = np.deg2rad(angle)
        
        # 计算模糊核的中心点
        center = kernel_size // 2
        
        # 计算模糊方向上的像素并应用高斯权重
        for i in range(kernel_size):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                # 计算高斯权重
                distance = np.sqrt((x - center)**2 + (y - center)**2)
                weight = np.exp(-(distance**2) / (2 * sigma**2))
                kernel[y, x] = weight
        
        # 归一化模糊核
        kernel = kernel / kernel.sum()
        
        # 应用模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image
    
    def _defocus_blur(self, image, radius=5):
        """添加散焦模糊
        
        Args:
            image: 输入图像
            radius: 圆盘半径（2~8）
            
        Returns:
            numpy.ndarray: 模糊后的图像
        """
        # 确保半径在有效范围内
        radius = max(2, min(8, radius))
        
        # 创建圆盘模糊核
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # 生成圆盘
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = np.sqrt((i - radius)**2 + (j - radius)**2)
                if distance <= radius:
                    kernel[i, j] = 1
        
        # 归一化模糊核
        kernel = kernel / kernel.sum()
        
        # 应用模糊
        blurred_image = cv2.filter2D(image, -1, kernel)
        return blurred_image
    
    def _random_blur_combination(self, image, max_combinations=2):
        """随机组合模糊退化
        
        Args:
            image: 输入图像
            max_combinations: 最大组合数量（1~3）
            
        Returns:
            numpy.ndarray: 退化后的图像
        """
        # 确保组合数量在有效范围内
        max_combinations = max(1, min(3, max_combinations))
        
        # 选择要应用的模糊类型
        blur_types = ['motion_blur', 'gaussian_blur', 'defocus_blur']
        selected_types = np.random.choice(blur_types, size=max_combinations, replace=False)
        
        result = image.copy()
        
        # 应用每种模糊类型
        for blur_type in selected_types:
            if blur_type == 'motion_blur':
                # 随机参数
                kernel_size = np.random.randint(10, 20)
                angle = np.random.randint(0, 360)
                sigma = np.random.uniform(1, 3)
                result = self._motion_blur(result, kernel_size=kernel_size, angle=angle, sigma=sigma)
            elif blur_type == 'gaussian_blur':
                # 随机参数（sigma=1~5）
                kernel_size = np.random.choice([3, 5, 7])
                sigma = np.random.uniform(1, 5)
                result = self._gaussian_blur(result, kernel_size=(kernel_size, kernel_size), sigma=sigma)
            elif blur_type == 'defocus_blur':
                # 随机参数（半径2~8）
                radius = np.random.randint(2, 9)
                result = self._defocus_blur(result, radius=radius)
        
        return result
    
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
