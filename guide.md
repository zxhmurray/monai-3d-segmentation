# MONAI 3D 医学影像分割 - 工程避坑与实操手册

> **编写本文档的目的**：让你在 M3 Mac 或普通 GPU 上跑通 3D 医学影像分割，不踩坑，不爆显存，不浪费时间。

---

## 模块一：环境配置与 Apple Silicon 算力排雷

### 1.1 核心依赖库及版本

| 库名 | 推荐版本 | 备注 |
|------|---------|------|
| `torch` | 2.4.x | **不要**指定 `--index-url cu118`，默认安装会自动识别 MPS |
| `monai` | 1.3+ | 核心框架 |
| `nibabel` | 5.0+ | 读写 .nii.gz |
| `SimpleITK` | 2.2+ | 图像预处理 |
| `numpy` | 1.24+ | 数值计算 |

### 1.2 macOS M1/M2/M3 安装命令（正确姿势）

```bash
# ❌ 错误示范：指定了 CUDA 版本，Mac 上根本用不了
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# ✅ 正确示范：不指定源，让 pip 自动安装 Mac 原生版本
conda create -n monai3d python=3.10 -y
conda activate monai3d
pip install torch torchvision
pip install monai nibabel SimpleITK numpy pandas scipy scikit-learn matplotlib pyyaml tqdm
```

### 1.3 验证 MPS 加速是否启用

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS可用: {torch.backends.mps.is_available()}'); print(f'设备: {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')"
```

**预期输出：**
```
PyTorch: 2.x.x
MPS可用: True
设备: mps
```

> 💡 **避坑提示**：如果 MPS 显示 `False`，说明 PyTorch 版本不对。删掉重装，不要指定任何 `--index-url`。

### 1.4 显存警告（Mac 用户必读）

⚠️ **硬件警告**：3D U-Net 的 128³ Patch 在 Mac MPS 上约占用 **4-6GB 显存**。M3 Mac 有统一内存，如果你同时开了其他大应用，**系统可能会直接 kill 掉进程**。

**保命三招：**
1. 训练前关闭所有不必要的程序
2. 从小 patch 开始测试：`--roi_size 64 64 64`
3. 遇到 OOM 直接回退到 CPU

### 1.5 数据存放路径（必须严格遵守）

```
D:/Murray/MONAI/           ← 项目根目录
└── rawdata/
    └── MSD_Spleen/        ← 必须叫这个名字！
        ├── imagesTr/      ← CT 原始图像（从官网下载的原始文件夹）
        └── labelsTr/      ← 分割标注（从官网下载的原始文件夹）
```

> 💡 **避坑提示**：MSD 官网下载的数据集解压后，文件夹里是 `imagesTr` 和 `labelsTr`，**不要**改成 `images` 和 `labels`，否则代码会找不到数据。

**正确下载方式：**
1. 去 http://medicaldecathlon.com/ 下载 Task09_Spleen.tar
2. 解压到 `rawdata/MSD_Spleen/`
3. 确保目录结构如上图所示

---

## 模块二：医学影像"人话"科普与可视化检查

### 2.1 推荐可视化工具

| 工具 | 平台 | 下载地址 | 推荐程度 |
|------|------|---------|---------|
| **ITK-SNAP** | Win/Mac/Linux | www.itksnap.org | ⭐⭐⭐⭐⭐ |
| **3D Slicer** | Win/Mac/Linux | www.slicer.org | ⭐⭐⭐⭐ |

**同时打开原图和 Label 的步骤（以 ITK-SNAP 为例）：**
1. 打开 ITK-SNAP
2. File → Open Main Image → 选择 `.nii.gz` 原始图像
3. File → Open Label → 选择 `.nii.gz` Label 文件
4. 点击 "Overlay" 或 "Label" 按钮切换显示
5. 拖动滑块可以一层层看 CT 切片

### 2.2 Spacing (体素间距) - 通俗解释

**人话版**：CT 扫描出来的图像，每个像素不是正方形的。机器扫描时，X/Y 方向可能是 0.7mm，但 Z 方向（层厚）可能是 5mm。这导致图像看起来"扁扁的"。

**为什么不能强行重采样到 (1.0, 1.0, 1.0)？**

> ⚠️ **避坑提示**：强行插值到完全等方体，会在 Z 方向"创造"大量不存在的像素。Mac 显存本来就不大，这样做既浪费算力，又可能引入人工伪影。

**推荐策略**：重采样到 `(1.5, 1.5, 2.0)` 或 `(1.0, 1.0, 2.0)`，保持 Z 方向层厚接近原始，节省显存。

### 2.3 Orientation (方向) - RAS+ 坐标系

**人话版**：医学影像有自己的一套坐标系统。CT 图像存储时，有的机器是"头朝上、脚朝下"存储的，有的相反。`RAS+` 是一种标准约定：
- **R** = Right（X轴朝右）
- **A** = Anterior（Y轴朝前）
- **S** = Superior（Z轴朝上）

训练前把方向统一成 RAS+，能避免模型学到"奇怪的方向感"。

---

## 模块三：标准操作流 SOP

### Step 1: 数据校验（Sanity Check）

**训练前必做！** 先看看数据到底长什么样：

```bash
cd D:/Murray/MONAI
python -c "
import nibabel as nib
import os

# 找到第一张图
img_dir = 'rawdata/MSD_Spleen/imagesTr'
label_dir = 'rawdata/MSD_Spleen/labelsTr'
files = sorted(os.listdir(img_dir))
first_file = files[0]

# 读取
img = nib.load(f'{img_dir}/{first_file}')
lbl = nib.load(f'{label_dir}/{first_file}')

print(f'文件: {first_file}')
print(f'图像 Shape: {img.shape}')
print(f'标注 Shape: {lbl.shape}')
print(f'图像 Spacing: {img.header.get_zooms()}')
print(f'标注唯一值: {nib.load(f\"{label_dir}/{first_file}\").get_fdata().flatten().max()}')
"
```

**预期输出（MSD Spleen）：**
```
文件: spleen_1.nii.gz
图像 Shape: (512, 512, ~200)
标注 Shape: (512, 512, ~200)
图像 Spacing: (0.7, 0.7, 5.0)
标注唯一值: 1.0
```

> 💡 **避坑提示**：标注唯一值必须是 `1.0`（前景）和 `0.0`（背景）。如果看到 `255` 或 `2`，说明标注不是 0/1 二值，后续训练会出问题！

### Step 2: 启动训练（防 OOM 保命版）

**M3 Mac 推荐配置（先用这套跑通）：**

```bash
python scripts/train.py \
    --data_dir rawdata/MSD_Spleen \
    --max_epochs 50 \
    --roi_size 96 96 96 \
    --batch_size 1
```

**如果你显存更大（16GB+），可以尝试：**

```bash
python scripts/train.py \
    --data_dir rawdata/MSD_Spleen \
    --max_epochs 50 \
    --roi_size 128 128 128 \
    --batch_size 1
```

**超参解释：**

| 参数 | 作用 | Mac 推荐值 | 说明 |
|------|------|-----------|------|
| `--roi_size` | 每次喂给模型的图像块大小 | 96 96 96 | 越小越省显存，越大上下文信息越丰富 |
| `--batch_size` | 每批样本数 | 1 | Mac 上只能设 1，多了必爆显存 |
| `--device` | 计算设备 | auto | 默认自动检测 MPS/CUDA，OOM 时手动设 `cpu` |

**为什么 batch_size 只能设 1？**

> ⚠️ **显存警告**：3D 卷积的中间激活值非常占显存。batch_size=2 时显存占用约是 batch_size=1 的 **1.8 倍**。M3 Mac 用户老老实实设 1，梯度累积来凑 batch size。

### Step 3: 训练监控

训练过程中会自动保存日志，用 TensorBoard 查看：

```bash
# 安装 tensorboard（如果没装）
pip install tensorboard

# 启动 TensorBoard
tensorboard --logdir=results/logs

# 浏览器打开 http://localhost:6006
```

**看什么指标：**
- `Dice`：越接近 1 越好，0.90+ 算及格
- `Loss`：应该持续下降，如果变成 NaN 要立即停止
- `learning_rate`：监控是否正常衰减

---

## 模块四：滑窗推理与指标解读

### 4.1 滑窗推理原理（通俗类比）

**想象你用手电筒在黑暗的房间里找东西：**

1. **手电筒光斑** = `roi_size`（比如 128×128×128）
2. **光斑照亮的区域** = 一次推理的区域
3. **每次移动的距离** = `stride`（步长）
4. **相邻光斑重叠** = `overlap`（重叠率）

**为什么要重叠？**

> 💡 **避坑提示**：如果手电筒紧挨着移动，边界处会有"接缝"。重叠就是为了让相邻区域有平滑过渡，**推荐 overlap 设为 0.5（50%）**。

**拼接伪影怎么消除？**
- 使用 **Gaussian 加权融合**：重叠区域中心权重高，边缘权重低
- 就像 Photoshop 的"羽化"效果，让拼接处更自然

### 4.2 跑分预期（MSD Spleen）

| 训练 Epoch | 预期 Dice | 状态 |
|-----------|----------|------|
| 0-10 | 0.30-0.50 | 刚起步 |
| 20-30 | 0.70-0.85 | 正常学习 |
| 50 | **0.90+** | 及格线 ✅ |
| 80-100 | **0.95+** | 优秀 ⭐ |

> 💡 **避坑提示**：Dice 超过 0.95 反而要警惕——可能是过拟合或者标注泄露。医学影像分割很少能稳定超过 0.97。

---

## 模块五：常见错误与排查路线图

### 症状 1：进程被 Killed 或 OOM 🚨

**错误表现：**
```
RuntimeError: CUDA out of memory
# 或者
Process killed (OOM)
```

**解药（按顺序尝试）：**

1. **减小 roi_size**（首选）：
   ```bash
   --roi_size 64 64 64
   ```

2. **切换到 CPU**（保底）：
   ```bash
   --device cpu
   ```
   CPU 训练会慢 10-20 倍，但一定能跑通。

3. **减小网络通道数**：
   编辑 `src/model_builder/config.py`，把 `SMALL_MODEL_CONFIG` 的 channels 改成 `(8, 16, 32, 64)`

### 症状 2：Loss 变成 NaN 或 Dice 一直是 0 🚨

**错误表现：**
```
loss: nan
# 或
Dice: 0.0000 (无论训练多久)
```

**解药：**

1. **检查标注数值**：
   ```python
   import nibabel as nib
   lbl = nib.load("rawdata/MSD_Spleen/labelsTr/spleen_1.nii.gz")
   print(f"唯一值: {set(lbl.get_fdata().flatten())}")
   ```
   应该是 `{0.0, 1.0}`，如果有 `255` 或其他值，说明标注需要预处理。

2. **检查归一化是否应用**：
   确保 Transform 里包含 `ScaleIntensityRanged`，把 CT 值缩放到 0-1 范围。

3. **降低学习率**：
   ```bash
   --lr 0.00005
   ```

### 症状 3：预测 Mask 与原图完全错位 🚨

**错误表现**：
- ITK-SNAP 里看，Mask 和原图对不上
- 预测的器官位置"飘"到了奇怪的地方

**解药：**

> ⚠️ **避坑提示**：这通常是推理阶段**遗忘重采样或方向变换**导致的。

确保推理时使用了和训练**完全相同的 Transform**：
- Spacing 重采样
- Orientation 归一化到 RAS+
- CT 值归一化

---

## 快速命令清单

```bash
# 1. 激活环境
conda activate monai3d

# 2. 进入项目目录
cd D:/Murray/MONAI

# 3. 数据校验
python -c "import nibabel as nib; ..."

# 4. M3 Mac 保命训练配置
python scripts/train.py \
    --data_dir rawdata/MSD_Spleen \
    --max_epochs 50 \
    --roi_size 96 96 96 \
    --batch_size 1

# 5. 评估
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pt \
    --data_dir rawdata/MSD_Spleen \
    --device auto

# 6. 单张预测
python scripts/predict.py \
    --model models/checkpoints/best_model.pt \
    --input 你的图像.nii.gz \
    --output results/predictions
```

---

**祝训练顺利！如果遇到文档没覆盖的错误，请把完整的错误信息贴出来问我。**
