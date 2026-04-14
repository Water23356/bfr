import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
import cv2

class DataProvider:
    def __init__(self, data_home='../data'):
        """初始化数据提供者
        
        Args:
            data_home: 数据集存储目录
        """
        self.data_home = data_home
        self.lfw_data = None
        self.degraded_data = None
    
    def load_lfw_dataset(self, resize=0.5, min_faces_per_person=0):
        """加载 LFW 数据集
        
        Args:
            resize: 图像缩放比例
            min_faces_per_person: 每个人至少需要的图像数量
            
        Returns:
            dict: 包含图像和标签的字典
        """
        print("正在加载 LFW 数据集...")
        
        try:
            self.lfw_data = fetch_lfw_people(
                data_home=self.data_home,
                resize=resize,
                min_faces_per_person=min_faces_per_person,
                download_if_missing=False
            )
            
            print(f"数据集加载成功!")
            print(f"图像数量: {self.lfw_data.images.shape[0]}")
            print(f"类别数量: {len(self.lfw_data.target_names)}")
            print(f"图像尺寸: {self.lfw_data.images.shape[1:3]}")
            
            return {
                'images': self.lfw_data.images,
                'targets': self.lfw_data.target,
                'target_names': self.lfw_data.target_names
            }
        except Exception as e:
            print(f"数据集加载失败: {e}")
            print("请确保 LFW 数据集已下载到指定目录")
            return None
    
    def load_degraded_dataset(self, degraded_dir, resize=0.5):
        """加载退化数据集
        
        Args:
            degraded_dir: 退化数据集目录
            resize: 图像缩放比例
            
        Returns:
            dict: 包含图像和标签的字典
        """
        print(f"正在加载退化数据集: {degraded_dir}")
        
        try:
            images = []
            targets = []
            target_names = []
            
            # 遍历目录结构
            person_dirs = [d for d in os.listdir(degraded_dir) if os.path.isdir(os.path.join(degraded_dir, d))]
            
            for person_idx, person_name in enumerate(person_dirs):
                person_dir = os.path.join(degraded_dir, person_name)
                
                # 读取该人物的所有图像
                for img_file in os.listdir(person_dir):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(person_dir, img_file)
                        
                        # 读取图像
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            # 调整图像大小
                            if resize != 1.0:
                                new_size = (int(img.shape[1] * resize), int(img.shape[0] * resize))
                                img = cv2.resize(img, new_size)
                            
                            # 归一化到 0-1
                            img = img / 255.0
                            
                            images.append(img)
                            targets.append(person_idx)
                            
                # 添加目标名称
                # 将下划线替换为空格，恢复原始人物名称格式
                original_name = person_name.replace('_', ' ')
                target_names.append(original_name)
            
            if not images:
                print("未找到图像文件")
                return None
            
            # 转换为 numpy 数组
            images = np.array(images)
            targets = np.array(targets)
            target_names = np.array(target_names)
            
            # 保存到实例变量
            self.degraded_data = {
                'images': images,
                'targets': targets,
                'target_names': target_names
            }
            
            print(f"数据集加载成功!")
            print(f"图像数量: {images.shape[0]}")
            print(f"类别数量: {len(target_names)}")
            print(f"图像尺寸: {images.shape[1:3]}")
            
            return self.degraded_data
        except Exception as e:
            print(f"数据集加载失败: {e}")
            return None
    
    def get_person_images(self, person_index, num_images=2, use_degraded=False):
        """获取指定人物的图像
        
        Args:
            person_index: 人物索引
            num_images: 需要的图像数量
            use_degraded: 是否使用退化数据集
            
        Returns:
            list: 图像列表
        """
        data = self.degraded_data if use_degraded else self.lfw_data
        
        if data is None:
            print("请先加载数据集")
            return None
        
        # 找到该人物的所有图像
        if use_degraded:
            # 退化数据集是字典格式
            person_images = data['images'][data['targets'] == person_index]
        else:
            # LFW 数据集是 Bunch 对象格式
            person_images = data.images[data.target == person_index]
        
        if len(person_images) < num_images:
            print(f"该人物只有 {len(person_images)} 张图像，少于请求的 {num_images} 张")
            return person_images[:len(person_images)]
        
        return person_images[:num_images]
    
    def get_random_person(self, min_faces=2, use_degraded=False):
        """获取随机人物的索引，确保至少有 min_faces 张图像
        
        Args:
            min_faces: 至少需要的图像数量
            use_degraded: 是否使用退化数据集
            
        Returns:
            int: 人物索引
        """
        data = self.degraded_data if use_degraded else self.lfw_data
        
        if data is None:
            print("请先加载数据集")
            return None
        
        # 统计每个人的图像数量
        if use_degraded:
            # 退化数据集是字典格式
            unique_targets, counts = np.unique(data['targets'], return_counts=True)
        else:
            # LFW 数据集是 Bunch 对象格式
            unique_targets, counts = np.unique(data.target, return_counts=True)
        
        valid_targets = unique_targets[counts >= min_faces]
        
        if len(valid_targets) == 0:
            print(f"没有人物拥有至少 {min_faces} 张图像")
            return None
        
        # 随机选择一个人物
        return np.random.choice(valid_targets)
