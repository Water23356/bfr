import os
import sys
import argparse
import numpy as np
from core.data_provider import DataProvider
from channels import DegradationChannel

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class DegradedDatasetGenerator:
    def __init__(self, output_dir='../data/degraded', data_provider=None):
        """初始化数据集生成器
        
        Args:
            output_dir: 输出目录
            data_provider: 外部传入的DataProvider对象，若为None则使用默认实例
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data_provider = data_provider if data_provider is not None else DataProvider()
    
    def generate(self, degradation_type=None, num_images=100, degradation_channel=None, **params):
        """生成退化数据集
        
        Args:
            degradation_type: 退化类型（当degradation_channel为None时使用）
            num_images: 生成的图像数量
            degradation_channel: 外部传入的退化通道对象，若为None则根据degradation_type创建
            **params: 退化参数
        """
        # 加载 LFW 数据集
        data = self.data_provider.load_lfw_dataset()
        if data is None:
            print("无法加载数据集，生成终止")
            return
        
        # 创建或使用外部传入的退化通道
        if degradation_channel is not None:
            # 使用外部传入的退化通道
            channel = degradation_channel
            # 生成输出目录名
            degradation_type = getattr(channel, '__class__', channel).__name__
        else:
            # 创建新的退化通道
            if degradation_type is None:
                print("必须指定degradation_type或传入degradation_channel")
                return
            channel = DegradationChannel(degradation_type, **params)
        
        # 创建输出目录
        degradation_dir = os.path.join(self.output_dir, degradation_type)
        os.makedirs(degradation_dir, exist_ok=True)
        
        # 生成退化图像
        print(f"开始生成 {degradation_type} 退化数据集...")
        print(f"目标数量: {num_images}")
        
        import cv2
        count = 0
        for i, (image, target) in enumerate(zip(data['images'], data['targets'])):
            if count >= num_images:
                break
            
            try:
                # 应用退化
                degraded = channel.process(image)
                
                # 保存图像
                person_name = data['target_names'][target].replace(' ', '_')
                person_dir = os.path.join(degradation_dir, person_name)
                os.makedirs(person_dir, exist_ok=True)
                
                image_path = os.path.join(person_dir, f"{person_name}_{i}.jpg")
                cv2.imwrite(image_path, degraded)
                
                count += 1
                if count % 10 == 0:
                    print(f"生成进度: {count}/{num_images}")
                    
            except Exception as e:
                print(f"处理图像 {i} 时出错: {e}")
                continue
        
        print(f"\n数据集生成完成！")
        print(f"生成了 {count} 张退化图像")
        print(f"保存目录: {degradation_dir}")
        return degradation_dir
    
    def generate_all_blur_types(self, num_images=50):
        """生成所有模糊类型的退化数据集
        
        Args:
            num_images: 每种类型生成的图像数量
        """
        # 定义模糊类型和参数
        blur_configs = [
            {'type': 'motion_blur', 'params': {'kernel_size': 15, 'angle': 45, 'sigma': 2}},
            {'type': 'gaussian_blur', 'params': {'sigma': 3}},
            {'type': 'defocus_blur', 'params': {'radius': 5}},
            {'type': 'random_blur', 'params': {'max_combinations': 2}}
        ]
        
        for config in blur_configs:
            print(f"\n=== 生成 {config['type']} 数据集 ===")
            self.generate(config['type'], num_images, **config['params'])

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成退化数据集')
    
    parser.add_argument('--degradation', type=str, default='random_blur',
                        choices=['motion_blur', 'gaussian_blur', 'defocus_blur', 'random_blur'],
                        help='退化类型')
    
    parser.add_argument('--num-images', type=int, default=100,
                        help='生成的图像数量')
    
    parser.add_argument('--output-dir', type=str, default='../data/degraded',
                        help='输出目录')
    
    parser.add_argument('--all', action='store_true',
                        help='生成所有模糊类型的数据集')
    
    # 退化参数
    parser.add_argument('--kernel-size', type=int, default=15,
                        help='运动模糊核大小')
    
    parser.add_argument('--angle', type=int, default=45,
                        help='运动模糊角度')
    
    parser.add_argument('--sigma', type=float, default=2,
                        help='高斯模糊标准差')
    
    parser.add_argument('--radius', type=int, default=5,
                        help='散焦模糊半径')
    
    parser.add_argument('--max-combinations', type=int, default=2,
                        help='随机组合最大数量')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    generator = DegradedDatasetGenerator(output_dir=args.output_dir)
    
    if args.all:
        generator.generate_all_blur_types(num_images=args.num_images)
    else:
        # 构建参数
        params = {}
        if args.degradation == 'motion_blur':
            params['kernel_size'] = args.kernel_size
            params['angle'] = args.angle
            params['sigma'] = args.sigma
        elif args.degradation == 'gaussian_blur':
            params['sigma'] = args.sigma
        elif args.degradation == 'defocus_blur':
            params['radius'] = args.radius
        elif args.degradation == 'random_blur':
            params['max_combinations'] = args.max_combinations
        
        generator.generate(args.degradation, num_images=args.num_images, **params)

if __name__ == "__main__":
    main()
