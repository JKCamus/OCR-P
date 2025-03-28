# imgocr 项目使用指南

本文档详细记录了如何设置、启动和使用 imgocr 项目，方便后续功能添加和维护。

## 目录

- [环境设置](#环境设置)
  - [创建虚拟环境](#创建虚拟环境)
  - [激活虚拟环境](#激活虚拟环境)
  - [安装依赖](#安装依赖)
- [使用方法](#使用方法)
  - [基本OCR识别](#基本ocr识别)
  - [图形界面演示](#图形界面演示)
  - [命令行批量处理](#命令行批量处理)
- [开发指南](#开发指南)
  - [项目结构](#项目结构)
  - [常见问题解决](#常见问题解决)
- [后续开发计划](#后续开发计划)

## 环境设置

### 创建虚拟环境

为了避免依赖冲突，建议使用虚拟环境运行项目。以下是创建虚拟环境的步骤：

```bash
# 进入项目目录
cd /path/to/imgocr

# 创建虚拟环境
python -m venv venv
```

### 激活虚拟环境

每次使用项目前，需要先激活虚拟环境：

```bash
# 在macOS/Linux上
source venv/bin/activate

# 在Windows上
venv\Scripts\activate
```

激活后，命令行前面会出现`(venv)`前缀，表示当前处于虚拟环境中。

### 安装依赖

首次设置项目时，需要安装所有依赖：

```bash
# 安装基本依赖
pip install -r requirements.txt

# 以开发模式安装imgocr包
pip install -e .

# 安装Gradio（用于图形界面演示）
pip install gradio
```

## 使用方法

### 基本OCR识别

可以通过Python代码直接使用imgocr进行OCR识别：

```python
from imgocr.ppocr_onnx import ImgOcr

# 初始化OCR模型（默认使用高效率模式）
model = ImgOcr(use_gpu=False, is_efficiency_mode=True)

# 对图片进行OCR识别
result = model.ocr("path/to/image.jpg")

# 打印识别结果
for item in result:
    print(item['text'])
```

参数说明：
- `use_gpu`: 是否使用GPU，默认False
- `is_efficiency_mode`: 是否使用高效率模型，默认True
  - True: 使用高效率模型(mobile，14MB)，速度更快，精度稍低
  - False: 使用高精度模型(server，207MB)，精度更高，速度较慢

### 图形界面演示

项目提供了基于Gradio的图形界面演示，可以直观地体验OCR功能：

```bash
# 确保已激活虚拟环境
source venv/bin/activate

# 运行Gradio演示
python examples/gradio_demo.py
```

运行后，浏览器会自动打开一个本地网页（通常是http://127.0.0.1:7861），提供以下功能：
- 上传图片进行OCR识别
- 选择高效率/高精度模型
- 查看文本识别结果和可视化结果
- 查看识别统计信息

### 命令行批量处理

imgocr提供了命令行工具，可以对整个文件夹的图片进行批量OCR处理：

```bash
# 确保已激活虚拟环境
source venv/bin/activate

# 运行命令行工具
imgocr --image_dir path/to/images
```

参数说明：
- `--image_dir`: 输入图片目录路径（必需）
- `--output_dir`: 输出OCR结果目录路径，默认为outputs
- `--chunk_size`: 分块大小，默认为10
- `--use_gpu`: 是否使用GPU，默认为False

## 开发指南

### 项目结构

主要文件和目录说明：

```
imgocr/
├── examples/             # 示例代码
│   ├── data/             # 示例图片
│   ├── ocr_demo.py       # 基本OCR示例
│   ├── gradio_demo.py    # 图形界面演示
│   └── ...
├── imgocr/               # 主要代码
│   ├── __init__.py       # 包初始化
│   ├── ppocr_onnx.py     # OCR核心实现
│   ├── utils.py          # 工具函数
│   ├── models/           # 模型文件
│   └── ...
├── tests/                # 测试代码
├── requirements.txt      # 依赖列表
├── setup.py              # 安装脚本
└── README.md             # 项目说明
```

### 常见问题解决

1. **导入错误**：如果遇到`ImportError: cannot import name 'ImgOcr' from 'imgocr'`，请确保使用正确的导入路径：
   ```python
   from imgocr.ppocr_onnx import ImgOcr
   ```

2. **图片路径错误**：确保图片路径正确，可以使用绝对路径或相对于当前工作目录的路径。

3. **数据类型溢出**：如果遇到`OverflowError: Python integer 255 out of bounds for int8`，这是因为在`utils.py`中创建空白图像时使用了`np.int8`类型。已修复为`np.uint8`。

4. **模型下载问题**：首次使用高精度模型时，会自动下载模型文件（约207MB），请确保网络连接正常。

## 项目特点

1. **轻量级部署**：无需安装paddlepaddle、paddleocr等深度学习库，仅需安装onnxruntime，即可使用OCR功能。
2. **高性能**：使用ONNX Runtime进行推理，速度快，资源占用少。
3. **易用性**：提供简单易用的API和命令行工具，方便集成到各种应用中。
4. **多场景支持**：支持文本检测、文本识别、文本方向分类等多种OCR任务。

## 模型文件管理

由于OCR模型文件较大（高精度模型超过100MB），不适合直接包含在Git仓库中，我们采用了自动下载机制：

1. **模型下载**：
   ```bash
   # 只下载高效率模型（移动端模型，约15MB）
   python -m imgocr.download_models
   
   # 或下载所有模型（包括高精度服务器模型，约200MB）
   python -m imgocr.download_models all
   ```

2. **模型存储位置**：所有模型文件将保存在`imgocr/models/`目录下。

3. **Git仓库管理**：
   - 模型文件已添加到`.gitignore`中，不会被Git跟踪
   - 推送到GitHub时不会包含这些大型模型文件
   - 用户首次使用时需要下载模型文件

## 后续开发计划

可以考虑添加以下功能：

1. **界面优化**：
   - 添加更多自定义选项（如检测阈值调整）
   - 支持批量处理功能
   - 添加结果导出功能

2. **模型增强**：
   - 添加更多语言支持
   - 优化小文本识别
   - 提供文档分析功能

3. **部署优化**：
   - 提供Docker部署方案
   - 添加API服务
   - 优化移动设备支持

---

## 快速启动命令备忘

```bash
# 激活环境并运行图形界面
source venv/bin/activate && python examples/gradio_demo.py

# 激活环境并运行基本OCR示例
source venv/bin/activate && python examples/ocr_demo.py
```
