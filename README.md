# Transformer 英德翻译

从零实现的Transformer模型，用于IWSLT2017英德翻译任务。

## 数据集

下载地址：https://huggingface.co/datasets/IWSLT/iwslt2017


## 环境配置

### 1. 创建虚拟环境

```bash
# 使用conda
conda create -n transformer python=3.8
conda activate transformer

# 或使用venv
python -m venv transformer_env
source transformer_env/bin/activate  # Linux/Mac
# 或
transformer_env\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 安装PyTorch (CPU版本)
pip install torch numpy matplotlib

# 或安装PyTorch (GPU版本，推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib

# 或直接使用requirements.txt
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 运行

```bash
python main.py
```

## 配置

在`main.py`中修改`config`字典：

- `num_heads`: 注意力头数（2/4/8）
- `num_epochs`: 训练轮数
- `dropout`: Dropout比率
- `initial_lr`: 初始学习率

## 输出

- `checkpoints/best_model.pt` - 最佳模型
- `checkpoints/training_curves.png` - 训练曲线
- `checkpoints/training_summary.txt` - 训练摘要

## 模型

- 6层Encoder + 6层Decoder
- 31.3M参数
- 线性学习率衰减（0.0001→0.00001）

## 硬件

- GPU: NVIDIA RTX 4090
- 显存: ~24GB
- 时间: ~7分钟/epoch

