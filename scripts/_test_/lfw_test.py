# lfw_test.py
import sklearn.datasets as datasets
import os

# 检查数据集是否已存在
data_dir = './data'
if os.path.exists(os.path.join(data_dir, 'lfw_home')):
    print("数据集已存在，正在加载...")
    lfw_images = datasets.fetch_lfw_people(data_home=data_dir, resize=0.5, download_if_missing=False)
    
    # 打印数据集的基本信息
    print(f"\n数据集图像张数: {lfw_images.images.shape[0]}")
    print(f"图像高度(H) x 宽度(W): {lfw_images.images.shape[1:]}")
    print(f"类别数量: {len(lfw_images.target_names)}")
    print(f"第一个人的名称: {lfw_images.target_names[0]}")
    print(f"前5个标签: {lfw_images.target[:5]}")
    print(f"前5个标签对应的名称: {lfw_images.target_names[lfw_images.target[:5]]}")
    
    print("\n数据集加载成功！")
else:
    print("数据集不存在，准备下载...")
    print("注意：LFW数据集约200MB，下载可能需要一些时间")
    print("\n或者您可以手动下载数据集：")
    print("1. 访问：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html")
    print("2. 下载数据集并解压到 ./data/lfw_home 目录")
    print("\n正在尝试自动下载...")
    
    try:
        lfw_images = datasets.fetch_lfw_people(data_home=data_dir, resize=0.5)
        print("\n数据集下载成功！")
        print(f"数据集图像张数: {lfw_images.images.shape[0]}")
        print(f"图像高度(H) x 宽度(W): {lfw_images.images.shape[1:]}")
        print(f"类别数量: {len(lfw_images.target_names)}")
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("请尝试手动下载数据集")