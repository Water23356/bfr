import os
import sys
import time
import json
import numpy as np

# 添加 scripts 目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_provider import DataProvider
from .evaluation import Evaluation

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

class BaseEvaluator:
    """基础评估器 - 直接使用通道对象"""
    def __init__(self, log_dir='../logs'):
        """初始化评估器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.data_provider = DataProvider()
        self.evaluator = Evaluation()
    
    def evaluate(self, degradation_channel, restoration_channel, recognition_channel, num_tests=5):
        """执行评估
        
        Args:
            degradation_channel: 退化通道对象
            restoration_channel: 修复通道对象
            recognition_channel: 识别通道对象
            num_tests: 测试数量
        """
        # 加载数据集
        print("正在加载 LFW 数据集...")
        data = self.data_provider.load_lfw_dataset()
        if data is None:
            print("无法加载数据集，评估终止")
            return
        
        # 创建日志文件
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'evaluation_{timestamp}.json')
        
        # 获取通道类型
        degradation_type = getattr(degradation_channel, 'degradation_type', type(degradation_channel).__name__)
        restoration_type = getattr(restoration_channel, 'restoration_type', type(restoration_channel).__name__)
        
        # 准备评估结果
        evaluation_results = {
            'timestamp': timestamp,
            'degradation': {
                'type': degradation_type,
                'params': degradation_channel.params
            },
            'restoration': {
                'type': restoration_type,
                'params': restoration_channel.params
            },
            'recognition': {
                'model': recognition_channel.model_name
            },
            'num_tests': num_tests,
            'tests': [],
            'average': {}
        }
        
        # 运行测试
        print(f"开始评估: {degradation_type} -> {restoration_type}")
        print(f"测试数量: {num_tests}")
        print("评估进度:")
        
        total_degraded_mse = 0
        total_degraded_psnr = 0
        total_degraded_ssim = 0
        total_restored_mse = 0
        total_restored_psnr = 0
        total_restored_ssim = 0
        total_mse_change = 0
        total_psnr_change = 0
        total_ssim_change = 0
        total_degraded_recognition = 0
        total_restored_recognition = 0
        valid_tests = 0
        
        for i in range(num_tests):
            print(f"  测试 {i+1}/{num_tests}...", end=' ')
            
            # 获取随机人物的图像
            person_index = self.data_provider.get_random_person(min_faces=2)
            if person_index is None:
                print("跳过")
                continue
            
            person_images = self.data_provider.get_person_images(person_index, num_images=2)
            if len(person_images) < 2:
                print("跳过")
                continue
            
            # 使用第一张图像作为原始图像
            original = person_images[0]
            
            try:
                # 应用退化
                degraded = degradation_channel.process(original)
                
                # 应用修复
                restored = restoration_channel.process(degraded)
                
                # 计算退化图像的质量指标
                degraded_mse = self.evaluator.calculate_mse(original, degraded)
                degraded_psnr = self.evaluator.calculate_psnr(original, degraded)
                degraded_ssim = self.evaluator.calculate_ssim(original, degraded)
                
                # 计算修复图像的质量指标
                restored_mse = self.evaluator.calculate_mse(original, restored)
                restored_psnr = self.evaluator.calculate_psnr(original, restored)
                restored_ssim = self.evaluator.calculate_ssim(original, restored)
                
                # 计算指标变化量
                mse_change = degraded_mse - restored_mse  # 正值表示改善
                psnr_change = restored_psnr - degraded_psnr  # 正值表示改善
                ssim_change = restored_ssim - degraded_ssim  # 正值表示改善
                
                # 计算退化图像识别率
                degraded_verified, degraded_distance = recognition_channel.verify(original, degraded)
                
                # 计算修复图像识别率
                restored_verified, restored_distance = recognition_channel.verify(original, restored)
                
                # 记录测试结果
                test_result = {
                    'test_id': i+1,
                    'person_index': int(person_index),
                    'degraded_metrics': {
                        'mse': float(degraded_mse),
                        'psnr': float(degraded_psnr),
                        'ssim': float(degraded_ssim)
                    },
                    'restored_metrics': {
                        'mse': float(restored_mse),
                        'psnr': float(restored_psnr),
                        'ssim': float(restored_ssim)
                    },
                    'metric_changes': {
                        'mse_change': float(mse_change),
                        'psnr_change': float(psnr_change),
                        'ssim_change': float(ssim_change)
                    },
                    'degraded_recognition': {
                        'verified': degraded_verified,
                        'distance': float(degraded_distance)
                    },
                    'restored_recognition': {
                        'verified': restored_verified,
                        'distance': float(restored_distance)
                    }
                }
                
                evaluation_results['tests'].append(test_result)
                
                # 累计值
                total_degraded_mse += degraded_mse
                total_degraded_psnr += degraded_psnr
                total_degraded_ssim += degraded_ssim
                total_restored_mse += restored_mse
                total_restored_psnr += restored_psnr
                total_restored_ssim += restored_ssim
                total_mse_change += mse_change
                total_psnr_change += psnr_change
                total_ssim_change += ssim_change
                total_degraded_recognition += 1 if degraded_verified else 0
                total_restored_recognition += 1 if restored_verified else 0
                valid_tests += 1
                
                print("完成")
                
            except Exception as e:
                print(f"失败: {e}")
                continue
        
        # 计算平均结果
        if valid_tests > 0:
            average_degraded_mse = total_degraded_mse / valid_tests
            average_degraded_psnr = total_degraded_psnr / valid_tests
            average_degraded_ssim = total_degraded_ssim / valid_tests
            average_restored_mse = total_restored_mse / valid_tests
            average_restored_psnr = total_restored_psnr / valid_tests
            average_restored_ssim = total_restored_ssim / valid_tests
            average_mse_change = total_mse_change / valid_tests
            average_psnr_change = total_psnr_change / valid_tests
            average_ssim_change = total_ssim_change / valid_tests
            average_degraded_recognition = (total_degraded_recognition / valid_tests) * 100
            average_restored_recognition = (total_restored_recognition / valid_tests) * 100
            
            evaluation_results['average'] = {
                'degraded_metrics': {
                    'mse': float(average_degraded_mse),
                    'psnr': float(average_degraded_psnr),
                    'ssim': float(average_degraded_ssim)
                },
                'restored_metrics': {
                    'mse': float(average_restored_mse),
                    'psnr': float(average_restored_psnr),
                    'ssim': float(average_restored_ssim)
                },
                'metric_changes': {
                    'mse_change': float(average_mse_change),
                    'psnr_change': float(average_psnr_change),
                    'ssim_change': float(average_ssim_change)
                },
                'degraded_recognition_rate': float(average_degraded_recognition),
                'restored_recognition_rate': float(average_restored_recognition),
                'valid_tests': valid_tests
            }
            
            # 保存日志文件
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            # 显示最终评估指标
            print("\n===== 最终评估指标 =====")
            print(f"有效测试数: {valid_tests}")
            print("\n退化图像指标:")
            print(f"  平均 MSE: {average_degraded_mse:.4f}")
            print(f"  平均 PSNR: {average_degraded_psnr:.2f} dB")
            print(f"  平均 SSIM: {average_degraded_ssim:.4f}")
            print("\n修复图像指标:")
            print(f"  平均 MSE: {average_restored_mse:.4f}")
            print(f"  平均 PSNR: {average_restored_psnr:.2f} dB")
            print(f"  平均 SSIM: {average_restored_ssim:.4f}")
            print("\n指标变化量:")
            print(f"  MSE 改善: {average_mse_change:.4f}")
            print(f"  PSNR 改善: {average_psnr_change:.2f} dB")
            print(f"  SSIM 改善: {average_ssim_change:.4f}")
            print("\n识别率:")
            print(f"  平均退化图像识别率: {average_degraded_recognition:.2f}%")
            print(f"  平均修复图像识别率: {average_restored_recognition:.2f}%")
            print(f"\n日志文件: {log_file}")
        else:
            print("\n没有有效的测试结果")
