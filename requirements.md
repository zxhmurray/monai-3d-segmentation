# Requirements Document

## 项目概述
- 项目名称: MONAI 3D 医学影像分割与分析系统
- 项目类型: 桌面应用 / Web 服务
- 核心功能: 3D 医学影像的分割、标注与分析

---

## 功能需求

### 1. 核心功能

#### 1.1 自动下载数据集
- 从公开医学影像数据集网站搜集项目中会用到的数据集
- 自动下载并保存到项目的 `rawdata` 文件夹中
- 数据集来源参考：
  - [MONAI Dataset](https://github.com/Project-MONAI/MONAI#datasets)
  - [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/)
  - [KiTS Kidney Tumor Segmentation Challenge](https://kits21.kits-challenge.org/)
  - [BTCV (Beyond The Cranial Vault) Abdomen CT Segmentation](https://www.synapse.org/#!Synapse:syn3193805)
  - [CHAOS Challenge (CT/MRI Liver and Tumor Segmentation)](https://chaos.grand-challenge.org/)
  - [MSD Task01_BrainTumor](http://medicaldecathlon.com/#tasks)
  - [MSD Task09_Spleen]

#### 1.2 3D 数据读取与标准化
- 支持读取标准医学影像格式（NIfTI, DICOM, MHD 等）
- 物理空间标准化：
  - Spacing 重采样：将不同源设备的图像重采样到统一分辨率（如 1.0×1.0×1.0 mm³）
  - Orientation 归一化：将图像方向统一归一化（如 RAS+ 标准）
  - Intensity 强度标准化：归一化到 [0, 1] 或 z-score 标准化

#### 1.3 3D 空间预处理与增强
- 预处理流水线（Pipeline）：
  - `ScaleIntensity`：强度裁剪与缩放
  - `RandCropByPosNegLabel`：随机裁剪（基于正负样本平衡）
  - `RandAffine`：随机仿射变换（平移、旋转、缩放）
  - `Rand3DElastic`：随机弹性形变（扩充 3D 空间特征）
- 数据验证与质量检查

#### 1.4 网络构建与训练
- 基于 MONAI 构建 3D U-Net 分割模型
  - 编码器-解码器架构
  - 跳跃连接（Skip Connections）
  - 可配置的网络深度与通道数
- 混合损失函数：
  - `DiceCELoss`（Dice Loss + Cross Entropy Loss）
  - 可选：`DiceLoss`, `FocalLoss`, `TverskyLoss`
- 训练配置：
  - 学习率、批大小、优化器（Adam/AdamW）
  - 学习率调度器（CosineAnnealing, StepLR）
  - 早停机制（Early Stopping）
  - 混合精度训练（AMP）

#### 1.5 滑窗推理与验证
- 滑窗推理（Sliding Window Inference）：
  - 解决 3D 医疗影像显存占用大的问题
  - 可配置窗口大小、步长（Stride）
  - 重叠处理（Overlap）与结果融合（Blend Mode）
- 验证指标：
  - Dice 相似系数（Dice Metric）作为核心指标
  - 可选：IoU (Jaccard), HD95 (Hausdorff Distance), ASD (Average Surface Distance)

#### 1.6 结构化结果输出与分析
- 输出内容：
  - 预测的 3D Mask 文件（NIfTI 格式）
  - 脏器体积计算结果（Volume in mm³ / cm³）
  - 各个病例的 Dice 评估报告（CSV/JSON 格式）
  - 可视化：2D 切片对比、3D 渲染图
- 结果保存路径：
  - 预测结果：`results/predictions/`
  - 评估报告：`results/reports/`
  - 模型权重：`models/checkpoints/`

---

### 2. 用户界面
- 命令行界面（CLI）用于快速实验
- 可选：Web 界面用于结果可视化

---

### 3. 数据处理
- 数据集路径管理
- 训练/验证/测试集划分
- 数据加载器（DataLoader）配置

---

### 4. 模型相关
- 模型保存与加载（Checkpoints）
- 迁移学习支持
- 预训练权重加载

---

### 5. 部署与性能
- GPU 加速支持（CUDA）
- 批量处理优化
- 内存管理（避免 OOM）

---

## 技术栈

### 框架与库
- **MONAI** (Medical Open Network for AI): 核心框架
- **PyTorch**: 深度学习后端
- **NiBabel**: NIfTI 格式读写
- **SimpleITK**: 医学影像通用处理
- **NumPy/Pandas**: 数据处理与分析
- **Matplotlib/Plotly**: 可视化

### 依赖服务
- Python 3.8+
- CUDA 11.8+ / CuDNN 8+
- 建议 GPU 显存 ≥ 8GB（用于 3D 分割）

---

## 非功能需求

### 性能要求
- 单样本推理时间：< 5 秒（典型 3D CT/MRI）
- 训练时间：单 epoch ≤ 10 分钟（MSD Spleen 数据集）

### 安全要求
- 输入数据校验
- 模型版本管理

### 兼容性
- Windows / Linux 双平台支持

---

## 项目结构

```
MONAI/
├── rawdata/              # 原始数据集
├── data/                 # 处理后的数据
├── models/               # 模型权重与检查点
│   └── checkpoints/
├── results/              # 推理结果与报告
│   ├── predictions/
│   └── reports/
├── scripts/              # 训练与推理脚本
├── configs/              # 配置文件
├── requirements.txt
└── README.md
```

---

## 里程碑

- [ ] M1: 环境搭建与数据集获取
- [ ] M2: 3D 数据读取与标准化
- [ ] M3: 预处理流水线实现
- [ ] M4: 3D U-Net 模型构建与训练
- [ ] M5: 滑窗推理与验证指标
- [ ] M6: 结果输出与分析报告
- [ ] M7: 完整流程联调与测试

---

## 待确认事项
- [ ] 具体使用哪个公开数据集进行开发测试（建议：MSD Spleen 或 KiTS）
- [ ] 是否需要支持 DICOM 格式直接读取
- [ ] 是否需要 Web 可视化界面
- [ ] 目标部署平台（本地/GPU 服务器/云端）
- [ ] 具体的硬件配置（GPU 型号与显存大小）
