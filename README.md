# MONAI 3D Medical Image Segmentation System

基于 MONAI 框架的 3D 医学影像分割系统，支持脾脏等器官的自动分割。

## 功能特性

- **3D 数据流水线**：NIfTI 图像加载、Spacing 重采样、Orientation 归一化
- **数据增强**：RandCropByPosNegLabel、RandAffine、RandFlip 等
- **3D U-Net 模型**：可配置的编码器-解码器架构
- **DiceCELoss 混合损失**：处理类别不平衡
- **滑窗推理**：支持大体积 3D 图像的显存优化推理
- **后处理**：连通域分析、体积计算、评估报告生成
- **FastAPI 服务化**：RESTful API 支持图像上传与推理
- **Web Dashboard**：可视化前端，支持拖拽上传、结果展示
- **Docker 部署**：支持 CPU/GPU 一键部署

## 项目结构

```
MONAI/
├── src/                      # 源代码
│   ├── data_pipeline/       # 数据流水线
│   │   ├── loaders.py       # NIfTI 读取
│   │   ├── resample.py      # 重采样
│   │   ├── normalizer.py    # 强度归一化
│   │   ├── transforms.py    # MONAI Transforms
│   │   └── datasets.py      # Dataset 封装
│   ├── model_builder/       # 模型构建
│   │   ├── unet.py          # 3D U-Net
│   │   └── config.py        # 网络配置
│   ├── training_engine/     # 训练引擎
│   │   ├── trainer.py        # 训练循环
│   │   └── loss.py          # 损失函数
│   └── evaluator/            # 推理与评估
│       ├── inference.py      # 滑窗推理
│       ├── postprocess.py    # 后处理
│       └── volume.py         # 体积计算
├── scripts/                  # 入口脚本
│   ├── train.py             # 训练
│   ├── predict.py           # 推理
│   ├── evaluate.py          # 评估
│   └── download_data.py     # 数据下载
├── api/                      # FastAPI 服务
│   ├── main.py              # 主应用
│   ├── inference.py         # 推理端点
│   ├── model_manager.py     # 模型管理
│   └── models.py            # 数据模型
├── web/                      # Web Dashboard
│   ├── src/
│   │   ├── components/      # React 组件
│   │   ├── pages/           # 页面
│   │   └── api/             # API 调用
│   └── package.json
├── configs/                  # 配置文件
├── deploy/                   # 部署配置
├── tests/                    # 单元测试
├── requirements.txt           # Python 依赖
└── guide.md                  # 使用指南

## 快速开始

### 1. 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# (可选) API 服务依赖
pip install -r requirements-api.txt

# (可选) 前端依赖
cd web && npm install
```

### 2. 准备数据

下载 MSD Spleen 数据集：

```bash
# 方法一：自动下载（如果源可用）
python scripts/download_data.py

# 方法二：手动下载
# 1. 访问 http://medicaldecathlon.com/
# 2. 下载 Task09_Spleen.tar
# 3. 解压到 rawdata/MSD_Spleen/
```

### 3. 训练模型

```bash
# CPU 训练
python scripts/train.py --data_dir rawdata/MSD_Spleen --max_epochs 100

# GPU 训练（推荐）
python scripts/train.py --data_dir rawdata/MSD_Spleen --max_epochs 100 --device cuda

# 使用自定义配置
python scripts/train.py --config configs/train_config.yaml
```

### 4. 推理预测

```bash
# 单图像推理
python scripts/predict.py \
    --model models/best_model.pt \
    --input rawdata/MSD_Spleen/images \
    --output results/predictions

# 带后处理
python scripts/predict.py \
    --model models/best_model.pt \
    --input path/to/image.nii.gz \
    --postprocess
```

### 5. 模型评估

```bash
python scripts/evaluate.py \
    --model models/best_model.pt \
    --data_dir rawdata/MSD_Spleen \
    --output_dir results/reports
```

## Web Dashboard

### 启动 API 服务

```bash
uvicorn api.main:app --reload --port 8000
```

### 启动前端

```bash
cd web
npm install
npm run dev
```

访问 http://localhost:3000

## Docker 部署

### CPU 模式

```bash
docker-compose up -d
```

### GPU 模式

```bash
docker-compose -f deploy/gpu-compose.yml up -d
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /health` | GET | 健康检查 |
| `GET /models` | GET | 列出可用模型 |
| `POST /inference/predict` | POST | 单图像推理 |
| `POST /inference/predict_batch` | POST | 批量推理 |
| `GET /inference/volume/{case_id}` | GET | 获取体积报告 |

完整 API 文档：http://localhost:8000/docs

## 配置参数

### 训练配置 (`configs/train_config.yaml`)

```yaml
training:
  max_epochs: 100
  batch_size: 2
  learning_rate: 1.0e-4
  val_interval: 5
  early_stopping:
    enabled: true
    patience: 20

model:
  model_size: medium  # small, medium, large
  in_channels: 1
  out_channels: 1

data:
  spatial_size: [128, 128, 128]
  pixdim: [1.0, 1.0, 1.0]
  ct_window: abdomen
```

### 推理配置 (`configs/inference_config.yaml`)

```yaml
inference:
  roi_size: [128, 128, 128]
  overlap: 0.5
  sw_batch_size: 4
  blend_mode: gaussian
  threshold: 0.5

postprocess:
  keep_largest: true
  min_volume: 500
  fill_holes: true
```

## 数据集

本项目使用 [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) 数据集中的 **Task09 Spleen** 数据集：

- **训练集**：41 个腹部 CT 图像
- **图像尺寸**：512 × 512 × ~90 体素
- **Spacing**：0.79 × 0.79 × 5.0 mm³
- **标注**：脾脏分割 mask

## 技术栈

- **框架**：PyTorch 2.x, MONAI 1.x
- **数据处理**：NiBabel, SimpleITK, NumPy
- **Web 框架**：FastAPI, React 18, TypeScript
- **容器化**：Docker, docker-compose, NVIDIA Docker

## 开发阶段

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 3D 数据流水线 | ✅ 完成 |
| Phase 2 | 网络构建与损失定义 | ✅ 完成 |
| Phase 3 | 训练引擎与验证逻辑 | ✅ 完成 |
| Phase 4 | 滑窗推理与后处理 | ✅ 完成 |
| Phase 5 | 服务化部署与前端交互 | ✅ 完成 |

## 注意事项

- **GPU 训练**：强烈建议使用 GPU，CPU 训练非常缓慢
- **数据安全**：原始医疗数据不会上传到 Git 仓库
- **模型权重**：训练出的模型文件不会上传到 Git 仓库
- **内存需求**：3D U-Net 训练需要至少 8GB 显存（medium size）

## 许可证

本项目仅供学习和研究使用。

## 致谢

- [MONAI](https://monai.io/) - Medical Open Network for AI
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) - 医学图像分割数据集
