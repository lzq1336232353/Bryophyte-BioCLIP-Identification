# Bryophyte-BioCLIP-Identification
A specialized bryophyte identification network built upon the BioCLIP 2.5 biological vision foundation model.

This repository provides the official implementation for benchmarking deep learning paradigms on fine-grained bryophyte taxonomy, evaluating the transition from traditional custom convolutional frameworks to multimodal biological foundation models under data-scarce conditions.

本项目提供了针对细粒度苔藓植物分类评估深度学习范式的官方实现。系统评估了在数据稀疏条件下，从传统自定义卷积框架向多模态生物大模型演进的技术表现。

---

## 📌 Project Overview | 项目简介

Comprehensive biodiversity monitoring of bryophytes is severely hindered by high phenotypic plasticity, morphological reduction, and miniature statures. This project addresses the "taxonomic impediment" by utilizing **BryophyteTBY**, a curated dataset featuring 1,197 research-grade images of seven keystone moss and liverwort species collected from the Tianbaoyan National Nature Reserve. 

We benchmark three technical paradigms across the repository:
1. **MOSSNet**: A replicated lightweight neural network utilizing a dual-branch multiscale attention design for extracting localized micro-morphology from scratch.
2. **ResNet-50**: A generic convolutional network leveraging global geometric features pre-trained on ImageNet.
3. **BioCLIP 2.5 (ViT-H/14)**: A domain-specific vision-language foundation model fine-tuned via a frozen-backbone linear probing strategy to unlock evolutionary and phylogenetic visual priors.

苔藓植物的自动化数字监测长期受到高表型可塑性与微观同源结构的制约。本项目基于自建的 **BryophyteTBY** 数据集（包含采集自天宝岩国家级自然保护区的7种核心苔藓植物、共1,197张高质图像），对三种深度学习范式进行了系统性基准测试：
1. **MOSSNet**：仿照文献复现的轻量化双分支多尺度注意力网络，从头训练以提取局部微观结构。
2. **ResNet-50**：经典的通用卷积网络，利用基于 ImageNet 预训练的泛化几何特征。
3. **BioCLIP 2.5 (ViT-H/14)**：领域特定的生物学视觉语言大模型，采用冻结主干网络的线性探针策略，解锁系统发育与演化视觉先验。

---

## 📊 Performance Benchmarks | 实验性能对比

| Model | Top-1 Acc (%) | F1 Score | FLOPs (G) | Params (M) | Paradigm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MOSSNet** | 57.56% | 0.5751 | 0.372 | 8.663 | Train from scratch |
| **ResNet-50** | 96.16% | 0.9615 | 4.132 | 23.522 | General Pre-trained |
| **BioCLIP 2.5** | **99.67%** | **0.9967** | 107.986 | 420.395 | Biological Foundation Model |

### Key Insights | 核心发现:
* **Foundation Model Dominance**: BioCLIP 2.5 yields near-perfect accuracy (99.67%), proving that domain-specific evolutionary priors bypass the mathematical bottlenecks of small-sample training.
* **From-Scratch Limitations**: Highly parameterized custom networks trained from scratch (MOSSNet) suffer severe feature entanglement under small-data constraints.
* **大模型压倒性优势**：BioCLIP 2.5 达到了 99.67% 的极高准确率，证明特定领域的生物学演化先验能彻底打破小样本训练的数学瓶颈。
* **从头训练的局限性**：在小样本数据约束下，从头训练的高度参数化自定义网络（MOSSNet）会面临严重的特征缠绕。

---

## 🖼️ Visualizations | 结果可视化

### 1. Dataset Exhibition | 数据集展示
![Dataset Exhibition](Figure%201_dataset_exhibition.jpg)
*Captures multi-habitat contexts, macro close-ups, and distinctive life-history growth periods (gametophytes vs. reproductive sporophytes) to ensure real-world robustness.*
涵盖了多生境背景、宏观微距、以及交替世代中不同的生育期特征（配子体与孢子体），确保模型具备应对现实野外环境的鲁棒性。

---

### 2. Error Analysis via Confusion Matrix | 混淆矩阵分析
![Confusion Matrix](Figure%202_Unified_Confusion_Matrix_Compact.jpg)
*Empirical heatmaps demonstrate that BioCLIP 2.5 successfully decouples historically challenging, morphologically convergent lineages (e.g., separating the thalloid liverworts Marchantia polymorpha and Reboulia hemisphaerica) that confound conventional convolutional filters.*
热力图表明，BioCLIP 2.5 成功解耦了在传统卷积滤波器中极易混淆的同源同塑性物种（如准确区分叶状体苔类 *Marchantia polymorpha* 与 *Reboulia hemisphaerica*）。

---

### 3. Latent Manifold Visualization | 潜空间流形分布
![t-SNE Manifold](Figure%203_comparative_tsne_3models.jpg)
*t-SNE dimensions demonstrate that leveraging specialised, taxonomic-aware pre-trained representations achieves a tightly clustered, highly polarized embedding space with distinct semantic boundaries.*
t-SNE 降维结果表明，利用领域特定的生物大模型表征能够使连续潜空间产生高度极化，拉开清晰的分类边界，呈现出紧凑的类内凝聚力。

---

## 📂 Repository Structure | 目录架构

```txt
├── Figure 1_dataset_exhibition.jpg            # Curation and pipeline preview
├── Figure 2_Unified_Confusion_Matrix_Compact.jpg # Heatmap for error vectors
├── Figure 3_comparative_tsne_3models.jpg      # 2D t-SNE dimension manifolds
├── README.md                                  # Project documentation
├── dataset.py                                 # Decoupled preprocessing and pipeline loaders for MOSSNet and ResNet-50
├── model.py                                   # Model structural architectures (MOSSNet)
├── train_mossnet.py                           # Training pipeline for MOSSNet from scratch
├── train_resnet50.py                          # Finetuning pipeline for ResNet-50
├── bioclip2-5.ipynb                           # Notebook for frozen-backbone linear probing
├── tsne_step1_extract_data.py                 # Script to extract penultimate high-d vectors
└── tsne_step2_plot.py                         # Script to project and render t-SNE maps

---

## ⚡ Quick Start | 快速开始

### 1. Environment Setup | 环境配置
Ensure you have Python 3.14+ and PyTorch installed. Install dependencies via:
确保环境搭载 Python 3.14+ 与 PyTorch，使用以下命令安装依赖：
```bash
pip install torch torchvision open_clip_torch matplotlib scikit-learn pandas
```

### 2. Dataset Preparation | 数据集准备
Organize your curated images under a standard directory layout corresponding to `dataset.py`:
根据 `dataset.py` 的读取逻辑，将清洗好的图像按照标准类别文件夹放入数据根目录：
```text
data/
  ├── Acrolejeunea_sandvicensis/
  ├── Atrichum_undulatum/
  └── ...
```

### 3. Model Training & Execution | 模型训练与运行

* **To train MOSSNet from scratch | 从头训练 MOSSNet**:
  ```bash
  python train_mossnet.py
  ```
* **To finetune ResNet-50 | 微调 ResNet-50**:
  ```bash
  python train_resnet50.py
  ```
* **To extract and plot t-SNE manifolds | 提取并绘制 t-SNE 特征空间**:
  ```bash
  python tsne_step1_extract_data.py
  python tsne_step2_plot.py
  ```
* **For BioCLIP 2.5 Linear Probing | 运行 BioCLIP 2.5 线性探针**:
  Open and execute the step-by-step pipeline inside `bioclip2-5.ipynb`.
  打开并逐步运行 `bioclip2-5.ipynb` 脚本。

---

## 📜 Citation & License | 引用与开源协议

The custom implementation code in this repository is released under the **MIT License**.

If you use our comparative pipeline or the structural findings in your research, please cite our study alongside the foundation works:

```txt
@article{Liu_Bryophyte_BioCLIP_2026,
  title={From Custom Convolutional Bottlenecks to Foundation Model Probing: A Comparative Analysis of Deep Learning Frameworks for Fine-Grained Bryophyte Identification in Protected Nature Reserves},
  author={Liu, Zi-Qi},
  journal={School of Life Sciences, Xiamen University},
  year={2026}
}

@software{Gu_BioCLIP_2.5_Huge_model,
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  publisher = {Hugging Face},
  title = {{BioCLIP 2.5 Huge}},
  year = {2026},
  url = {[https://huggingface.co/imageomics/bioclip-2.5-vith14](https://huggingface.co/imageomics/bioclip-2.5-vith14)}
}
```
