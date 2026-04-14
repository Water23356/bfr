import os
import cv2
import numpy as np
from deepface import DeepFace

class RecognitionChannel:
    """人脸识别通道"""
    def __init__(self, model_name='VGG-Face', model_dir=None):
        """初始化识别通道
        
        Args:
            model_name: 人脸识别模型名称
            model_dir: 模型存储目录
        """
        # 使用绝对路径确保模型目录位置正确
        if model_dir is None:
            # 计算项目根目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 从 channels 目录向上三级到达项目根目录
            project_root = os.path.abspath(os.path.join(current_dir, '../..'))
            model_dir = os.path.join(project_root, 'models')
        
        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)
        
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
        # 使用绝对路径确保临时目录位置正确
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '../..'))
        temp_dir = os.path.join(project_root, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 确保图像是 0-255 的 uint8 格式
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # 保存图像
        image_path = os.path.join(temp_dir, filename)
        cv2.imwrite(image_path, image)
        return image_path
