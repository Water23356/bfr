import os
import cv2  # OpenCV 图像处理库
import numpy as np
from .base_restoration_channel import BaseRestorationChannel

# 尝试导入 PyTorch
torch_available = False
try:
    import torch
    from .nafnet_model import load_nafnet_model
    torch_available = True
except ImportError:
    pass

class NAFNetChannel(BaseRestorationChannel):
    """NAFNet 修复通道"""
    def __init__(self, model_path=None, device='cuda' if torch_available and torch.cuda.is_available() else 'cpu'):
        """初始化 NAFNet 通道
        
        Args:
            model_path: 模型权重路径
            device: 运行设备
        """
        super().__init__(model_path=model_path, device=device)
        self.model_path = model_path
        
        # 尝试使用默认模型路径
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '../..'))
            default_model_path = os.path.join(project_root, 'models', 'nafnet.pth')
            if os.path.exists(default_model_path):
                self.model_path = default_model_path
                print(f"使用默认模型权重: {self.model_path}")
        
        self.device = device
        self.model = None
        
        # 加载模型
        if torch_available:
            try:
                self.model = load_nafnet_model(self.model_path)
                self.model.to(device)
            except Exception as e:
                print(f"加载 NAFNet 模型失败: {e}")
                print("注意：NAFNet 模型需要预训练权重才能获得最佳效果")
        else:
            print("PyTorch 未安装，NAFNet 通道将使用原始图像")
    
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
        
        # 记录原始尺寸
        original_shape = image.shape
        
        # 如果 PyTorch 不可用或模型未加载，返回原始图像
        if not torch_available or self.model is None:
            return image
        
        # 处理灰度图像
        is_gray = len(image.shape) == 2
        if is_gray:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        try:
            # 转换为张量
            img_tensor = self._preprocess(image)
            
            # 推理
            with torch.no_grad():
                output_tensor = self.model(img_tensor)
            
            # 转换回 numpy 数组
            restored = self._postprocess(output_tensor)
            
            # 调整回原始尺寸
            restored = cv2.resize(restored, (original_shape[1], original_shape[0]))
            
            # 如果原始图像是灰度图，转换回灰度图
            if is_gray:
                restored = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            
            return restored
        except Exception as e:
            print(f"NAFNet 处理失败: {e}")
            return image
    
    def _preprocess(self, image):
        """预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 归一化到 [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) * 2.0
        
        # 转换为张量
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess(self, tensor):
        """后处理张量
        
        Args:
            tensor: 模型输出张量
            
        Returns:
            numpy.ndarray: 后处理后的图像
        """
        # 转换为 numpy 数组
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 反归一化
        image = (image / 2.0) + 0.5
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        
        return image
