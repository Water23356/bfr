import os
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from deepface import DeepFace

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class Evaluation:
    def __init__(self, model_dir='../models'):
        """初始化评估模块
        
        Args:
            model_dir: 模型存储目录
        """
        # 设置 DEEPFACE_HOME 环境变量
        os.environ['DEEPFACE_HOME'] = model_dir
        self.model_dir = model_dir
    
    def calculate_mse(self, original, restored):
        """计算均方误差 (MSE)
        
        Args:
            original: 原始图像
            restored: 修复后的图像
            
        Returns:
            float: MSE 值
        """
        # 确保图像是 0-255 的 uint8 格式
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        if restored.max() <= 1.0:
            restored = (restored * 255).astype(np.uint8)
        
        # 计算 MSE
        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        return mse
    
    def calculate_psnr(self, original, restored):
        """计算峰值信噪比 (PSNR)
        
        Args:
            original: 原始图像
            restored: 修复后的图像
            
        Returns:
            float: PSNR 值
        """
        # 确保图像是 0-255 的 uint8 格式
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        if restored.max() <= 1.0:
            restored = (restored * 255).astype(np.uint8)
        
        # 计算 PSNR
        psnr = peak_signal_noise_ratio(original, restored)
        return psnr
    
    def calculate_ssim(self, original, restored):
        """计算结构相似性指数 (SSIM)
        
        Args:
            original: 原始图像
            restored: 修复后的图像
            
        Returns:
            float: SSIM 值
        """
        # 确保图像是 0-255 的 uint8 格式
        if original.max() <= 1.0:
            original = (original * 255).astype(np.uint8)
        if restored.max() <= 1.0:
            restored = (restored * 255).astype(np.uint8)
        
        # 计算 SSIM
        ssim = structural_similarity(original, restored, data_range=255)
        return ssim
    
    def save_temp_image(self, image, filename):
        """保存临时图像
        
        Args:
            image: 图像数据
            filename: 文件名
            
        Returns:
            str: 图像路径
        """
        import cv2
        temp_dir = '../temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 保存图像
        image_path = os.path.join(temp_dir, filename)
        cv2.imwrite(image_path, image)
        return image_path
    
    def calculate_recognition_rate(self, image1, image2, model_name='VGG-Face'):
        """计算两张图像的识别率
        
        Args:
            image1: 第一张图像
            image2: 第二张图像
            model_name: 人脸识别模型名称
            
        Returns:
            bool: 是否识别为同一个人
            float: 相似度距离
        """
        try:
            # 保存临时图像
            img1_path = self.save_temp_image(image1, 'img1.jpg')
            img2_path = self.save_temp_image(image2, 'img2.jpg')
            
            # 使用 DeepFace 验证
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                enforce_detection=True
            )
            
            return result['verified'], result['distance']
        except Exception as e:
            print(f"识别失败: {e}")
            return False, float('inf')
    
    def evaluate_restoration(self, original, degraded, restored, model_name='VGG-Face'):
        """评估图像修复效果
        
        Args:
            original: 原始图像
            degraded: 退化后的图像
            restored: 修复后的图像
            model_name: 人脸识别模型名称
            
        Returns:
            dict: 评估结果
        """
        # 计算图像质量指标
        mse = self.calculate_mse(original, restored)
        psnr = self.calculate_psnr(original, restored)
        ssim = self.calculate_ssim(original, restored)
        
        # 计算退化图像识别率
        degraded_verified, degraded_distance = self.calculate_recognition_rate(original, degraded, model_name)
        
        # 计算修复图像识别率
        restored_verified, restored_distance = self.calculate_recognition_rate(original, restored, model_name)
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'degraded_recognition': {
                'verified': degraded_verified,
                'distance': degraded_distance
            },
            'restored_recognition': {
                'verified': restored_verified,
                'distance': restored_distance
            }
        }
    
    def print_evaluation_results(self, results):
        """打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print("\n===== 图像修复评估结果 =====")
        print(f"MSE: {results['mse']:.4f}")
        print(f"PSNR: {results['psnr']:.2f} dB")
        print(f"SSIM: {results['ssim']:.4f}")
        print(f"退化图像识别率: {'成功' if results['degraded_recognition']['verified'] else '失败'}")
        print(f"退化图像相似度: {results['degraded_recognition']['distance']:.4f}")
        print(f"修复图像识别率: {'成功' if results['restored_recognition']['verified'] else '失败'}")
        print(f"修复图像相似度: {results['restored_recognition']['distance']:.4f}")
