import os
import sys
import numpy as np
from data_provider import DataProvider
from image_degradation import ImageDegradation
from image_restoration import ImageRestoration
from evaluation import Evaluation

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class MainEvaluation:
    def __init__(self):
        """初始化主评估类"""
        self.data_provider = DataProvider()
        self.degrader = ImageDegradation()
        self.restorer = ImageRestoration()
        self.evaluator = Evaluation()
    
    def run_evaluation(self, num_tests=5):
        """运行完整的评估流程
        
        Args:
            num_tests: 测试次数
        """
        # 加载 LFW 数据集
        data = self.data_provider.load_lfw_dataset()
        if data is None:
            print("无法加载数据集，评估终止")
            return
        
        # 定义退化和修复算法
        degradation_methods = [
            {'type': 'gaussian_noise', 'params': {'sigma': 25}},
            {'type': 'salt_and_pepper', 'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
            {'type': 'motion_blur', 'params': {'kernel_size': 15, 'angle': 45}},
            {'type': 'gaussian_blur', 'params': {'kernel_size': (5, 5), 'sigma': 2}},
            {'type': 'downsample', 'params': {'scale': 0.5}}
        ]
        
        restoration_methods = [
            {'type': 'bilateral', 'params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}},
            {'type': 'non_local_means', 'params': {'h': 10}},
            {'type': 'sharpen', 'params': {'amount': 1.0}}
        ]
        
        # 运行测试
        all_results = []
        
        for i in range(num_tests):
            print(f"\n===== 测试 {i+1}/{num_tests} =====")
            
            # 获取随机人物的图像
            person_index = self.data_provider.get_random_person(min_faces=2)
            if person_index is None:
                continue
            
            person_images = self.data_provider.get_person_images(person_index, num_images=2)
            if len(person_images) < 2:
                continue
            
            # 使用第一张图像作为原始图像
            original = person_images[0]
            
            # 对每种退化方法进行测试
            for deg_method in degradation_methods:
                print(f"\n退化方法: {deg_method['type']}")
                
                # 应用退化
                degraded = self.degrader.apply_degradation(
                    original, 
                    deg_method['type'], 
                    **deg_method['params']
                )
                
                # 对每种修复方法进行测试
                for rest_method in restoration_methods:
                    print(f"修复方法: {rest_method['type']}")
                    
                    # 应用修复
                    restored = self.restorer.apply_restoration(
                        degraded, 
                        rest_method['type'], 
                        **rest_method['params']
                    )
                    
                    # 评估修复效果
                    results = self.evaluator.evaluate_restoration(
                        original, 
                        degraded, 
                        restored
                    )
                    
                    # 打印结果
                    self.evaluator.print_evaluation_results(results)
                    
                    # 保存结果
                    all_results.append({
                        'test': i+1,
                        'person_index': person_index,
                        'degradation': deg_method['type'],
                        'restoration': rest_method['type'],
                        'results': results
                    })
        
        # 计算平均结果
        if all_results:
            self.calculate_average_results(all_results)
    
    def calculate_average_results(self, all_results):
        """计算平均评估结果
        
        Args:
            all_results: 所有测试结果
        """
        print("\n===== 平均评估结果 =====")
        
        # 按退化方法分组
        by_degradation = {}
        for result in all_results:
            degradation = result['degradation']
            if degradation not in by_degradation:
                by_degradation[degradation] = []
            by_degradation[degradation].append(result['results'])
        
        # 计算每种退化方法的平均结果
        for degradation, results in by_degradation.items():
            print(f"\n退化方法: {degradation}")
            
            # 初始化累计值
            total_mse = 0
            total_psnr = 0
            total_ssim = 0
            total_degraded_recognition = 0
            total_restored_recognition = 0
            count = len(results)
            
            # 累计值
            for res in results:
                total_mse += res['mse']
                total_psnr += res['psnr']
                total_ssim += res['ssim']
                total_degraded_recognition += 1 if res['degraded_recognition']['verified'] else 0
                total_restored_recognition += 1 if res['restored_recognition']['verified'] else 0
            
            # 计算平均值
            avg_mse = total_mse / count
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            avg_degraded_recognition = (total_degraded_recognition / count) * 100
            avg_restored_recognition = (total_restored_recognition / count) * 100
            
            # 打印平均结果
            print(f"平均 MSE: {avg_mse:.4f}")
            print(f"平均 PSNR: {avg_psnr:.2f} dB")
            print(f"平均 SSIM: {avg_ssim:.4f}")
            print(f"平均退化图像识别率: {avg_degraded_recognition:.2f}%")
            print(f"平均修复图像识别率: {avg_restored_recognition:.2f}%")

if __name__ == "__main__":
    print("===== 图像修复评估系统 =====")
    
    evaluator = MainEvaluation()
    evaluator.run_evaluation(num_tests=3)
    
    print("\n===== 评估完成 =====")
