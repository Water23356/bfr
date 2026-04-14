import os
import sys
import numpy as np
# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli.generate_degraded_dataset import DegradedDatasetGenerator
from core.data_provider import DataProvider
from channels.gaussian_blur_channel import GaussianBlurChannel

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def test_generate_degraded_dataset():
    """测试生成退化数据集"""
    print("=== 测试生成退化数据集 ===")
    
    # 创建数据集生成器
    generator = DegradedDatasetGenerator(output_dir='../data/test_degraded')
    
    # 测试生成高斯模糊数据集
    degradation_dir = generator.generate(
        degradation_type='gaussian_blur',
        num_images=10,
        sigma=2
    )
    
    print(f"生成的数据集目录: {degradation_dir}")
    assert os.path.exists(degradation_dir), "数据集目录未创建"
    
    # 检查是否生成了图像
    person_dirs = [d for d in os.listdir(degradation_dir) if os.path.isdir(os.path.join(degradation_dir, d))]
    assert len(person_dirs) > 0, "未生成人物目录"
    
    # 检查每个人物目录是否有图像
    for person_dir in person_dirs:
        img_files = [f for f in os.listdir(os.path.join(degradation_dir, person_dir)) if f.endswith('.jpg')]
        assert len(img_files) > 0, f"{person_dir} 目录中没有图像"
    
    print("✓ 生成退化数据集测试通过")
    return degradation_dir

def test_load_degraded_dataset(degraded_dir):
    """测试加载退化数据集"""
    print("\n=== 测试加载退化数据集 ===")
    
    # 创建数据提供者
    data_provider = DataProvider()
    
    # 加载退化数据集
    data = data_provider.load_degraded_dataset(degraded_dir)
    assert data is not None, "退化数据集加载失败"
    
    # 检查数据结构
    assert 'images' in data, "数据中缺少 images 字段"
    assert 'targets' in data, "数据中缺少 targets 字段"
    assert 'target_names' in data, "数据中缺少 target_names 字段"
    
    # 检查数据类型
    assert isinstance(data['images'], np.ndarray), "images 不是 numpy 数组"
    assert isinstance(data['targets'], np.ndarray), "targets 不是 numpy 数组"
    assert isinstance(data['target_names'], np.ndarray), "target_names 不是 numpy 数组"
    
    # 检查数据维度
    assert data['images'].ndim == 3, "images 维度不正确"
    assert data['targets'].ndim == 1, "targets 维度不正确"
    assert data['target_names'].ndim == 1, "target_names 维度不正确"
    
    print("✓ 加载退化数据集测试通过")

def test_external_channel_and_provider():
    """测试使用外部通道和数据提供者"""
    print("\n=== 测试使用外部通道和数据提供者 ===")
    
    # 创建自定义数据提供者
    custom_provider = DataProvider(data_home='../data')
    
    # 创建自定义退化通道
    custom_channel = GaussianBlurChannel(sigma=3)
    
    # 创建数据集生成器
    generator = DegradedDatasetGenerator(
        output_dir='../data/test_external',
        data_provider=custom_provider
    )
    
    # 使用外部通道生成数据集
    degradation_dir = generator.generate(
        degradation_channel=custom_channel,
        num_images=5
    )
    
    print(f"使用外部通道生成的数据集目录: {degradation_dir}")
    assert os.path.exists(degradation_dir), "外部通道生成的数据集目录未创建"
    
    # 检查是否生成了图像
    person_dirs = [d for d in os.listdir(degradation_dir) if os.path.isdir(os.path.join(degradation_dir, d))]
    assert len(person_dirs) > 0, "外部通道未生成人物目录"
    
    print("✓ 使用外部通道和数据提供者测试通过")

def test_get_person_images():
    """测试获取人物图像"""
    print("\n=== 测试获取人物图像 ===")
    
    # 创建数据提供者
    data_provider = DataProvider()
    
    # 加载原始 LFW 数据集
    lfw_data = data_provider.load_lfw_dataset()
    assert lfw_data is not None, "原始 LFW 数据集加载失败"
    
    # 测试获取原始数据集人物图像
    person_idx = data_provider.get_random_person()
    assert person_idx is not None, "无法获取随机人物"
    
    person_images = data_provider.get_person_images(person_idx, num_images=2)
    assert len(person_images) >= 1, "无法获取人物图像"
    
    print("✓ 获取原始数据集人物图像测试通过")

def main():
    """主测试函数"""
    try:
        # 运行所有测试
        degraded_dir = test_generate_degraded_dataset()
        test_load_degraded_dataset(degraded_dir)
        test_external_channel_and_provider()
        test_get_person_images()
        
        print("\n🎉 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
