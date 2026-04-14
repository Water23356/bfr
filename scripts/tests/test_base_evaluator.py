import os
import sys

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BaseEvaluator
from channels import (
    GaussianBlurChannel,
    RichardsonLucyChannel,
    RecognitionChannel
)

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def test_base_evaluator():
    """测试基础评估器"""
    print("===== 测试基础评估器 =====")
    
    # 创建通道对象
    degradation_channel = GaussianBlurChannel(sigma=3)
    restoration_channel = RichardsonLucyChannel(iterations=30)
    recognition_channel = RecognitionChannel(model_name='VGG-Face')
    
    # 创建评估器
    evaluator = BaseEvaluator()
    
    # 执行评估
    evaluator.evaluate(degradation_channel, restoration_channel, recognition_channel, num_tests=2)

if __name__ == "__main__":
    test_base_evaluator()
