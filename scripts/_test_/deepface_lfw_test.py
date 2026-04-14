import os
import sys
import numpy as np
from deepface import DeepFace
import sklearn.datasets as datasets

# 设置默认编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 设置 DEEPFACE_HOME 环境变量，使用项目本地目录的模型
os.environ['DEEPFACE_HOME'] = os.path.join(os.path.dirname(__file__), 'models')
print(f"已设置 DEEPFACE_HOME 环境变量为: {os.environ['DEEPFACE_HOME']}")

# 模型目录
model_dir = os.path.join(os.environ['DEEPFACE_HOME'], '.deepface', 'weights')
print(f"模型目录: {model_dir}")
print()

# 检查模型文件是否存在
def check_models():
    # 人脸识别至少需要 vgg_face_weights.h5
    required_models = ['vgg_face_weights.h5']
    # 属性分析需要的模型
    optional_models = ['age_model_weights.h5', 'gender_model_weights.h5', 'facial_expression_model_weights.h5', 'race_model_single_batch.h5']
    
    missing_required = []
    missing_optional = []
    
    for model_name in required_models:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            missing_required.append(model_name)
    
    for model_name in optional_models:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            missing_optional.append(model_name)
    
    if missing_required:
        print("缺少以下必要模型文件:")
        for model_name in missing_required:
            print(f"- {model_name}")
        print("请确保模型文件已下载到上述目录")
        return False
    else:
        print("所有必要的模型文件都已存在")
        
        if missing_optional:
            print("缺少以下可选模型文件:")
            for model_name in missing_optional:
                print(f"- {model_name}")
            print("属性分析功能可能会受到限制")
        
        return True

# 加载 LFW 数据集
def load_lfw_dataset():
    print("正在加载 LFW 数据集...")
    data_dir = './data'
    
    try:
        lfw_images = datasets.fetch_lfw_people(data_home=data_dir, resize=0.5, download_if_missing=False)
        print(f"数据集加载成功！")
        print(f"图像数量: {lfw_images.images.shape[0]}")
        print(f"类别数量: {len(lfw_images.target_names)}")
        return lfw_images
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("请确保 LFW 数据集已下载到 ./data 目录")
        return None

# 保存图像到临时文件
def save_temp_image(image, filename):
    import cv2
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    # 将归一化的图像转换为 0-255 的 uint8 格式
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # 保存图像
    image_path = os.path.join(temp_dir, filename)
    cv2.imwrite(image_path, image)
    return image_path

# 测试人脸识别
def test_face_recognition(lfw_images):
    print("\n开始测试人脸识别...")
    
    # 选择前 5 个人的图像进行测试
    num_test = 5
    correct_count = 0
    total_count = 0
    
    for i in range(num_test):
        # 找到同一个人的所有图像
        person_idx = lfw_images.target[i]
        person_name = lfw_images.target_names[person_idx]
        person_images = lfw_images.images[lfw_images.target == person_idx]
        
        if len(person_images) >= 2:
            # 保存两张同一人的图像
            img1_path = save_temp_image(person_images[0], f"{person_name}_1.jpg")
            img2_path = save_temp_image(person_images[1], f"{person_name}_2.jpg")
            
            print(f"\n测试 {person_name} (图像数量: {len(person_images)})")
            
            try:
                # 使用 DeepFace 验证两张图像是否属于同一个人
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name='VGG-Face',
                    enforce_detection=True
                )
                
                print(f"验证结果: {result['verified']}")
                print(f"计算距离: {result['distance']:.4f}")
                print(f"判定阈值: {result['threshold']}")
                print(f"识别模型: {result['model']}")
                print(f"检测后端: {result['detector_backend']}")
                print(f"距离度量: {result['similarity_metric']}")
                
                if result['verified']:
                    correct_count += 1
                total_count += 1
                
            except Exception as e:
                print(f"验证失败: {e}")
    
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        print(f"\n测试完成!")
        print(f"测试总数: {total_count}")
        print(f"正确数: {correct_count}")
        print(f"准确率: {accuracy:.2f}%")
    else:
        print("\n没有足够的图像进行测试")

# 测试人脸属性分析
def test_face_attributes(lfw_images):
    print("\n开始测试人脸属性分析...")
    
    # 检查哪些模型可用
    available_actions = []
    
    # 检查年龄模型
    if os.path.exists(os.path.join(model_dir, 'age_model_weights.h5')):
        available_actions.append('age')
    
    # 检查性别模型
    if os.path.exists(os.path.join(model_dir, 'gender_model_weights.h5')):
        available_actions.append('gender')
    
    # 检查情绪模型
    if os.path.exists(os.path.join(model_dir, 'emotion_model_weights.h5')):
        available_actions.append('emotion')
    
    # 检查种族模型
    if os.path.exists(os.path.join(model_dir, 'race_model_weights.h5')):
        available_actions.append('race')
    
    if not available_actions:
        print("没有可用的属性分析模型")
        return
    
    print(f"可用的属性分析: {', '.join(available_actions)}")
    
    # 选择前 3 个人的图像进行测试
    num_test = 3
    
    for i in range(num_test):
        # 保存图像
        image = lfw_images.images[i]
        person_name = lfw_images.target_names[lfw_images.target[i]]
        img_path = save_temp_image(image, f"{person_name}_attr.jpg")
        
        print(f"\n分析 {person_name} 的属性")
        
        try:
            # 使用 DeepFace 分析人脸属性
            result = DeepFace.analyze(
                img_path=img_path,
                actions=available_actions,
                enforce_detection=True
            )
            
            # 打印分析结果
            if 'age' in available_actions:
                print(f"年龄: {result[0]['age']} 岁")
            
            if 'gender' in available_actions:
                print(f"性别: {result[0]['dominant_gender']}")
            
            if 'emotion' in available_actions:
                print(f"情绪: {result[0]['dominant_emotion']}")
            
            if 'race' in available_actions:
                print(f"种族: {result[0]['dominant_race']}")
            
        except Exception as e:
            print(f"分析失败: {e}")

if __name__ == "__main__":
    print("===== DeepFace LFW 测试 =====")
    
    # 检查模型文件
    if not check_models():
        sys.exit(1)
    
    # 加载 LFW 数据集
    lfw_images = load_lfw_dataset()
    if lfw_images is None:
        sys.exit(1)
    
    # 测试人脸识别
    test_face_recognition(lfw_images)
    
    # 测试人脸属性分析
    test_face_attributes(lfw_images)
    
    print("\n===== 测试结束 =====")
