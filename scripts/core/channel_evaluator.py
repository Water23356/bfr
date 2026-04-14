import os
import sys

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_evaluator import BaseEvaluator
from channels import (
    GaussianNoiseChannel,
    SaltAndPepperChannel,
    MotionBlurChannel,
    GaussianBlurChannel,
    DefocusBlurChannel,
    DownsampleChannel,
    CompositeDegradationChannel,
    RecognitionChannel,
    BilateralFilterChannel,
    NonLocalMeansChannel,
    SharpenChannel,
    WienerFilterChannel,
    RichardsonLucyChannel,
    SuperResolutionChannel,
    NAFNetChannel,
    CompositeRestorationChannel
)

class ChannelEvaluator:
    """通道评估器 - 解析通道文本标签"""
    def __init__(self, log_dir='../logs'):
        """初始化评估器
        
        Args:
            log_dir: 日志保存目录
        """
        self.base_evaluator = BaseEvaluator(log_dir=log_dir)
    
    def create_degradation_channel(self, degradation_type, **params):
        """创建退化通道
        
        Args:
            degradation_type: 退化类型字符串
            **params: 退化参数
            
        Returns:
            BaseDegradationChannel: 退化通道对象
        """
        if degradation_type == 'gaussian_noise':
            return GaussianNoiseChannel(**params)
        elif degradation_type == 'salt_and_pepper':
            return SaltAndPepperChannel(**params)
        elif degradation_type == 'motion_blur':
            return MotionBlurChannel(**params)
        elif degradation_type == 'gaussian_blur':
            return GaussianBlurChannel(**params)
        elif degradation_type == 'defocus_blur':
            return DefocusBlurChannel(**params)
        elif degradation_type == 'downsample':
            return DownsampleChannel(**params)
        else:
            raise ValueError(f"不支持的退化类型: {degradation_type}")
    
    def create_restoration_channel(self, restoration_type, **params):
        """创建修复通道
        
        Args:
            restoration_type: 修复类型字符串
            **params: 修复参数
            
        Returns:
            BaseRestorationChannel: 修复通道对象
        """
        if restoration_type == 'bilateral':
            return BilateralFilterChannel(**params)
        elif restoration_type == 'non_local_means':
            return NonLocalMeansChannel(**params)
        elif restoration_type == 'sharpen':
            return SharpenChannel(**params)
        elif restoration_type == 'wiener':
            return WienerFilterChannel(**params)
        elif restoration_type == 'richardson_lucy':
            return RichardsonLucyChannel(**params)
        elif restoration_type == 'super_resolution':
            return SuperResolutionChannel(**params)
        elif restoration_type == 'nafnet':
            return NAFNetChannel(**params)
        else:
            raise ValueError(f"不支持的修复类型: {restoration_type}")
    
    def evaluate(self, degradation_type, restoration_type, recognition_model='VGG-Face', num_tests=5, degradation_params=None, restoration_params=None):
        """执行评估
        
        Args:
            degradation_type: 退化类型字符串
            restoration_type: 修复类型字符串
            recognition_model: 人脸识别模型名称
            num_tests: 测试数量
            degradation_params: 退化参数
            restoration_params: 修复参数
        """
        # 处理默认参数
        if degradation_params is None:
            degradation_params = {}
        if restoration_params is None:
            restoration_params = {}
        
        # 创建通道
        degradation_channel = self.create_degradation_channel(degradation_type, **degradation_params)
        restoration_channel = self.create_restoration_channel(restoration_type, **restoration_params)
        recognition_channel = RecognitionChannel(model_name=recognition_model)
        
        # 执行评估
        self.base_evaluator.evaluate(degradation_channel, restoration_channel, recognition_channel, num_tests=num_tests)
