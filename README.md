# 基于 OWSM v4 与 Conv-Adapter PEFT 的多语种语种识别系统

本项目是《语音识别》课程综合项目，实现一个面向中文、英文、法语、日语、韩语五个语种的语音语种识别系统。系统输入一段语音后，输出最可能的语种以及五个语种的概率分布。

> 项目任务：语种识别（Language Identification）  
> 项目地址：https://github.com/jiangtdi/language_id_project

## 1. 项目概述

本项目使用 OWSM-CTC v4 作为预训练语音编码器，在其后加入 Conv-Adapter 参数高效适配模块、Temporal Attention Pooling 时序注意力池化和 Angular Margin 分类头，实现五语种分类。

项目重点不是语音转文字，而是判断一段语音属于哪一种语言。该能力可作为多语种语音识别、语音翻译、智能客服分流和语音数据整理的前置模块。

## 2. 支持语种

当前支持五类语种：

| 类别 | 语种 |
|---|---|
| Chinese | 中文 |
| English | 英文 |
| French | 法语 |
| Japanese | 日语 |
| Korean | 韩语 |

## 3. 技术路线

整体流程如下：

```text
输入语音
  ↓
统一为单声道 16kHz
  ↓
音量归一化、短音频补齐、长音频窗口切分
  ↓
OWSM-CTC v4 预训练语音编码器提取帧级语音特征
  ↓
Conv-Adapter PEFT 进行任务适配
  ↓
Temporal Attention Pooling 聚合时序特征
  ↓
Angular Margin 分类头输出五语种概率
  ↓
返回预测语种、置信度和概率分布
```

核心设计：

- **OWSM-CTC v4**：多语种预训练语音基础模型，负责提取高层语音特征。
- **冻结主干**：不全量微调 1B 级模型，降低显存和训练成本。
- **Conv-Adapter PEFT**：只训练轻量适配模块，使预训练特征更适合语种识别。
- **Temporal Attention Pooling**：自动关注更有语种区分度的时间片段。
- **Angular Margin 分类头**：增强不同语种在特征空间中的类别间隔。
- **域适配训练**：加入 VoxLingua107 数据，提高跨数据集泛化能力。

## 4. 近年技术依据

- [OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning](https://arxiv.org/abs/2506.00338)
- [OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification](https://aclanthology.org/2024.acl-long.549/)
- [Convolution-Augmented Parameter-Efficient Fine-Tuning for Speech Recognition](https://www.isca-archive.org/interspeech_2024/kim24s_interspeech.html)

## 5. 实验结果

### 5.1 内部测试集

- 测试样本：750 条
- 每类样本：150 条
- Test Accuracy：**100.00%**

| 语种 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Chinese | 1.0000 | 1.0000 | 1.0000 | 150 |
| English | 1.0000 | 1.0000 | 1.0000 | 150 |
| French | 1.0000 | 1.0000 | 1.0000 | 150 |
| Japanese | 1.0000 | 1.0000 | 1.0000 | 150 |
| Korean | 1.0000 | 1.0000 | 1.0000 | 150 |

### 5.2 VoxLingua107 外部测试集

- 测试样本：10000 条
- 每类样本：2000 条
- External Test Accuracy：**99.29%**

| 语种 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Chinese | 0.9940 | 0.9880 | 0.9910 | 2000 |
| English | 0.9925 | 0.9910 | 0.9917 | 2000 |
| French | 0.9995 | 0.9975 | 0.9985 | 2000 |
| Japanese | 0.9847 | 0.9960 | 0.9903 | 2000 |
| Korean | 0.9940 | 0.9920 | 0.9930 | 2000 |

说明：以上结果只代表当前五个目标语种和当前测试集条件下的表现，不等同于所有真实场景都能达到相同准确率。

## 6. 项目结构

```text
language_id_project/
├── src/                              # 核心模型、音频处理和工具代码
│   ├── audio_dataset.py              # 音频读取、重采样、切窗、增强
│   ├── logit_calibration.py          # logits 偏置校准工具
│   ├── owsm_adapter_lid_model.py     # OWSM + Adapter + Attention Pooling + 分类头
│   └── owsm_local_model.py           # 本地 OWSM 模型路径检查
├── backend/                          # FastAPI 推理后端
│   ├── main.py                       # 后端入口
│   └── inference_service.py          # 模型加载和推理封装
├── frontend/                         # React + Vite 前端演示界面
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
├── tests/                            # 核心模块测试
├── train_owsm_adapter_lid.py         # 模型训练脚本，支持断点续训和坏音频跳过
├── evaluate_owsm_adapter_lid.py      # 内部测试评估脚本
├── evaluate_external_audio_dir.py    # 外部测试评估脚本，支持断点续测
├── predict_owsm_adapter_lid.py       # 单条音频预测脚本
├── download_owsm_v4_model.py         # OWSM v4 模型下载脚本
├── download_fleurs_subset.py         # FLEURS 数据下载脚本
├── download_voxlingua_samples.py     # VoxLingua107 数据下载脚本
├── prepare_voxlingua_domain_split.py # VoxLingua107 域适配划分脚本
├── make_data1_subset.py              # 生成云服务器训练子集脚本
├── requirements.txt                  # Python 依赖
├── README.md                         # 项目说明
└── outputs/                          # 评估报告和图表输出
```

## 7. 大文件说明

以下目录或文件体积较大，不建议直接提交到普通 Git 仓库，已在 `.gitignore` 中排除：

- `owsm_ctc_v4_1B/`：OWSM v4 本地预训练模型目录。
- `data/`：训练集、验证集、测试集和外部测试数据。
- `checkpoints/*.pt`：训练好的模型权重和训练断点，当前最佳模型约 4GB。
- `frontend/node_modules/`：前端依赖目录。
- `outputs/external_voxlingua_predictions.csv`：外部测试逐条预测结果。

如果需要复现运行，请按照后续步骤重新下载模型和数据，或手动把本地权重文件放回对应目录。

## 8. 环境安装

建议 Python 版本：3.10 或 3.11。

安装 Python 依赖：

```powershell
pip install -r requirements.txt
```

如果 ESPnet 相关依赖未完整安装，可补充执行：

```powershell
pip install espnet espnet_model_zoo sentencepiece typeguard humanfriendly pyyaml
```

如果使用 GPU，请确保 PyTorch 与 CUDA 版本匹配。

## 9. 下载 OWSM v4 本地模型

推荐使用 HuggingFace CLI 下载：

```powershell
hf download espnet/owsm_ctc_v4_1B --repo-type model --local-dir .\owsm_ctc_v4_1B
```

也可以运行项目脚本：

```powershell
python download_owsm_v4_model.py
```

下载完成后，项目根目录应存在：

```text
owsm_ctc_v4_1B/
├── data/
├── exp/
├── meta.yaml
└── README.md
```

## 10. 数据准备

下载并整理 FLEURS 五语种数据：

```powershell
python download_fleurs_subset.py
```

下载并整理 VoxLingua107 外部测试数据：

```powershell
python download_voxlingua_samples.py
python prepare_voxlingua_domain_split.py
```

如果需要在云服务器上训练较小子集：

```powershell
python make_data1_subset.py
```

## 11. 训练模型

运行训练脚本：

```powershell
python train_owsm_adapter_lid.py
```

训练完成后会生成：

```text
checkpoints/best_owsm_adapter_lid.pt
checkpoints/label_map.json
outputs/owsm_adapter_training_curve.png
```

训练脚本支持：

- OWSM 主干冻结。
- Conv-Adapter PEFT 训练。
- 梯度累积。
- 标签平滑。
- 学习率调度。
- Early Stopping。
- 坏音频跳过。
- epoch 级断点续训。

## 12. 评估模型

内部测试：

```powershell
python evaluate_owsm_adapter_lid.py
```

外部 VoxLingua107 测试：

```powershell
python evaluate_external_audio_dir.py
```

外部测试耗时较长，脚本会保存逐条预测结果，支持中断后继续。

## 13. 单条音频预测

在 `predict_owsm_adapter_lid.py` 中设置音频路径后运行：

```powershell
python predict_owsm_adapter_lid.py
```

适合快速测试单条音频是否能被正确识别。

## 14. 启动 Web 演示系统

### 14.1 启动后端

```powershell
python backend/main.py
```

后端启动时会加载 OWSM v4 和训练好的分类模型，首次启动较慢是正常现象。

### 14.2 启动前端开发服务

另开一个终端：

```powershell
cd frontend
npm install
npm run dev
```

浏览器访问：

```text
http://127.0.0.1:5173
```

### 14.3 构建前端并由后端托管

```powershell
cd frontend
npm install
npm run build
cd ..
python backend/main.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```
