 # Bryophyte-BioCLIP-Identification
A pilot study exploring the use of BioCLIP 2.5 and conventional deep-learning models for bryophyte image classification.

This repository contains the code comparing three deep-learning approaches for fine-grained bryophyte image classification under a small-data setting.

---

## Project Overview

Comprehensive biodiversity monitoring of bryophytes is severely hindered by high phenotypic plasticity, morphological reduction, and miniature statures. This pilot project uses **BryophyteTBY**, a curated dataset of 1,197 images representing seven moss and liverwort species from Tianbaoyan National Nature Reserve, to explore the feasibility of image-based bryophyte identification.

We benchmark three technical paradigms across the repository:
1. **MOSSNet**: A replicated lightweight neural network utilizing a dual-branch multiscale attention design for extracting localized micro-morphology from scratch.
2. **ResNet-50**: A generic convolutional network leveraging global geometric features pre-trained on ImageNet.
3. **BioCLIP 2.5 (ViT-H/14)**: A biology-focused vision-language foundation model evaluated here using a frozen-backbone linear-probing approach.

---

## Performance Benchmarks

| Model | Top-1 Acc (%) | F1 Score | FLOPs (G) | Params (M) | Paradigm |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MOSSNet** | 57.56% | 0.5751 | 0.372 | 8.663 | Train from scratch |
| **ResNet-50** | 96.16% | 0.9615 | 4.132 | 23.522 | General Pre-trained |
| **BioCLIP 2.5** | **99.67%** | **0.9967** | 107.986 | 420.395 | Biological Foundation Model |

### Key Insights
* **BioCLIP performance**: BioCLIP 2.5 achieved the highest accuracy in the current experimental split (99.67%), suggesting that pretrained biological representations may be useful for this small bryophyte dataset.
* **Training from scratch**: MOSSNet performed substantially worse than the two pretrained models in the current experiment, suggesting that training from scratch may be less effective with this limited dataset.
---

## Visualizations

### 1. Dataset Exhibition
![Dataset Exhibition](Figure%201_dataset_exhibition.jpg)
*Captures multi-habitat contexts, macro close-ups, and distinctive life-history growth periods (gametophytes vs. reproductive sporophytes) to ensure real-world robustness.*

---

### 2. Error Analysis via Confusion Matrix
![Confusion Matrix](Figure%202_Unified_Confusion_Matrix_Compact.jpg)
*The confusion matrix shows that BioCLIP 2.5 correctly separated most images from the seven target species in the current test split, including morphologically similar taxa such as Marchantia polymorpha and Reboulia hemisphaerica.*

---

### 3. Latent Manifold Visualization
![t-SNE Manifold](Figure%203_comparative_tsne_3models.jpg)
*The t-SNE visualization shows clearer class separation for ResNet-50 and particularly BioCLIP 2.5 than for MOSSNet in this dataset. As t-SNE is primarily a visualization tool, these patterns are interpreted qualitatively rather than as evidence of taxonomic relationships.*

---

## Limitations

This project was developed as a course-based pilot study using a relatively small dataset of seven species. The reported performance reflects the current train/validation/test split and should not be interpreted as evidence of general performance across bryophytes.
Because some images may originate from the same observation, locality, or similar imaging conditions, potential data leakage or non-independent samples have not yet been fully evaluated. A more rigorous future evaluation would use group-based splitting by observation or collection event, additional species, and independent external test data.

---

## Repository Structure

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

## Quick Start
```
### 1. Environment Setup
Install dependencies via:
```bash
pip install torch torchvision open_clip_torch matplotlib scikit-learn pandas
```

### 2. Dataset Preparation
Organize your curated images under a standard directory layout corresponding to `dataset.py`:
```text
data/
  ├── Acrolejeunea_sandvicensis/
  ├── Atrichum_undulatum/
  └── ...
```

### 3. Model Training & Execution

* **To train MOSSNet from scratch**:
  ```bash
  python train_mossnet.py
  ```
* **To finetune ResNet-50**:
  ```bash
  python train_resnet50.py
  ```
* **To extract and plot t-SNE manifolds**:
  ```bash
  python tsne_step1_extract_data.py
  python tsne_step2_plot.py
  ```
* **For BioCLIP 2.5 Linear Probing**:
  Open and execute the step-by-step pipeline inside `bioclip2-5.ipynb`.

---

## Citation & License

This repository documents a course-based pilot project conducted at Xiamen University in 2026. The custom code is released under the MIT License.
BioCLIP 2.5 and other pretrained models used in this project should be cited according to their original publications and repositories.
