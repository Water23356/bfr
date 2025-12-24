# 任务包：基于最小二乘方滤波的 SRGAN 模糊人脸复原与识别（M2 包）

| 编号 | 任务名 | 交付物 | 人日 | 依赖 |
|---|---|---|---|---|
| T1 | 数据准备 | CelebA5k/LFW 原图 + 运动/散焦模糊核生成脚本 + 划分 JSON | 1 | — |
| T2 | 最小二乘方滤波模块 | `utils/filters.py`：频域反卷积 + γ 调参脚本 | 2 | T1 |
| T3 | SRGAN 生成器 | `models/srgan.py：Generator`（16 ResBlock + 2 Sub-pixel）类 | 3 | — |
| T4 | SRGAN 判别器 | `models/srgan.py：Discriminator`（VGG-like）类 | 2 | — |
| T5 | 损失函数 | `losses/`： perceptual(VGG19) + adv + L1 加权工厂 | 2 | T3,T4 |
| T6 | 数据集类 | `data/dataset.py`：LSRGANDataset（返回模糊图+清晰图） | 1 | T1 |
| T7 | 训练脚本 | `scripts/02_train_m2.py`：渐进训练、tensorboard、ckpt | 3 | T3,T5,T6 |
| T8 | 推理脚本 | `scripts/infer_m2.py`：单图/批量去模糊 + PSNR/SSIM 输出 | 1 | T7 |
| T9 | FaceNet 后端 | `utils/fnet.py`：128-D 特征提取 + 库模板比对函数 | 2 | — |
| T10 | 评估脚本 | `scripts/03_eval_m2.py`：复原指标 + 识别 Acc/AUC/耗时 | 1 | T8,T9 |
| T11 | 消融实验 | 报告：γ 值 / 损失权重 / 超分倍数 三组对比 | 2 | T10 |

>T2/T3/T4/T9 可同时开工
