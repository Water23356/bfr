import sys
import argparse
import os

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ChannelEvaluator

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='图像修复评估系统')
    
    # 退化通道参数
    parser.add_argument('--degradation', type=str, default='gaussian_noise',
                        choices=['gaussian_noise', 'salt_and_pepper', 'motion_blur', 'gaussian_blur', 'defocus_blur', 'downsample'],
                        help='退化处理类型')
    
    # 退化通道参数
    parser.add_argument('--degradation-params', type=str, default='',
                        help='退化处理参数，格式：param1=value1,param2=value2')
    
    # 修复通道参数
    parser.add_argument('--restoration', type=str, default='bilateral',
                        choices=['bilateral', 'non_local_means', 'sharpen', 'wiener', 'richardson_lucy', 'super_resolution', 'nafnet'],
                        help='图像修复类型')
    
    # 修复通道参数
    parser.add_argument('--restoration-params', type=str, default='',
                        help='图像修复参数，格式：param1=value1,param2=value2')
    
    # 识别通道参数
    parser.add_argument('--recognition-model', type=str, default='VGG-Face',
                        help='人脸识别模型名称')
    
    # 测试数量
    parser.add_argument('--num-tests', type=int, default=5,
                        help='测试数量')
    
    return parser.parse_args()

def parse_params(param_str):
    """解析参数字符串
    
    Args:
        param_str: 参数字符串，格式：param1=value1,param2=value2
        
    Returns:
        dict: 参数字典
    """
    params = {}
    if param_str:
        for param in param_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                # 尝试转换为数字
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                params[key.strip()] = value
    return params

def main():
    """主函数"""
    args = parse_arguments()
    
    # 解析参数
    degradation_params = parse_params(args.degradation_params)
    restoration_params = parse_params(args.restoration_params)
    
    # 执行评估
    evaluator = ChannelEvaluator()
    evaluator.evaluate(
        degradation_type=args.degradation,
        restoration_type=args.restoration,
        recognition_model=args.recognition_model,
        num_tests=args.num_tests,
        degradation_params=degradation_params,
        restoration_params=restoration_params
    )

if __name__ == "__main__":
    main()
