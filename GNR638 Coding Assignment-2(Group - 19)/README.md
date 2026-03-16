# Deep Representation Analysis and Transfer Learning Experiments

## Overview

This project performs a comprehensive study of **transfer learning, representation quality, and robustness of convolutional neural network backbones**. Multiple pretrained architectures are evaluated across several experimental setups to understand how representations evolve during training and how effectively they transfer to downstream tasks.

The experiments include:

- **Linear Probe Transfer**
- **Fine-Tuning Strategies**
- **Few-Shot Learning Analysis**
- **Corruption Robustness Evaluation**
- **Layer-Wise Feature Probing**

These experiments analyze:

- Transferability of pretrained representations
- Effects of fine-tuning strategies
- Data efficiency of models
- Robustness to corrupted inputs
- Evolution of semantic features across network depth

The backbone models used in this project are:

- **ResNet50**
- **DenseNet121**
- **EfficientNet-B0**

All models are implemented using the **timm PyTorch library**.

---

# Dataset

The dataset consists of **image classification data organized into class folders**.

## Expected Dataset Structure

```bash
train_data/
│
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── class_2/
│   ├── img1.jpg
│   ├── img2.jpg
│
...
```

## Mounting Dataset in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/train_data.zip -d /content/
```

---

# Project Structure

```bash
Project/
│
├── LinearProbeTransfer/
│   ├── results/
│   │   ├── resnet50/
│   │   ├── densenet121/
│   │   ├── efficientnet_b0/
│   │
│   ├── LinearProbeTransfer.py
│   ├── evaluate.py
│
├── Fine-Tuning_Strategies/
│   ├── resnet50/
│   ├── densenet121/
│   ├── efficientnet_b0/
│
├── FewShot_Analysis/
│
├── Robustness_Results/
│
├── Representation_Analysis/
│
└── notebook.ipynb
```

---

# Dependencies

The project relies on the following libraries.

## Deep Learning

- **PyTorch** – model training and tensor computations
- **timm** – pretrained CNN architectures
- **torchvision** – dataset loading and image transformations

## Data Analysis

- **NumPy** – numerical computations
- **scikit-learn** – PCA, logistic regression, and evaluation metrics
- **pandas** – structured data handling

## Visualization

- **Matplotlib** – plotting training curves
- **Seaborn** – visualization of confusion matrices

## Utilities

- **tqdm** – progress bars
- **THOP** – model complexity profiling

## Installation

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm thop
```

---

# Reproducibility

To ensure reproducibility, a fixed random seed is used across experiments.

```python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

# Experiment 1: Linear Probe Transfer

## Objective

Evaluate the **quality of pretrained representations** by freezing the backbone and training only a **linear classifier** on top of extracted features.

If the representations are strong, the classifier should achieve high accuracy without updating backbone weights.

---

## Configuration

Specify dataset and result directories before running the experiment.

```python
DATA_DIR = "path_to_training_dataset"
RESULT_DIR = "path_to_store_results"
```

---

## Output Structure

```bash
LinearProbeTransfer/
│
├── results/
│   ├── resnet50/
│   │   ├── log/
│   │   ├── plots/
│   │   │   ├── accuracy_curve.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── feature_pcs.png
│   │   │   ├── feature_tsne.png
│   │   │
│   │   ├── metrics.txt
│   │   ├── model.pth
│   │
│   ├── densenet121/
│   ├── efficientnet_b0/
```

---

## Generated Outputs

- Accuracy curves
- Confusion matrices
- PCA feature visualization
- t-SNE feature visualization
- Saved linear classifier weights

---

# Experiment 2: Fine-Tuning Strategies

## Objective

Compare different **fine-tuning strategies** and analyze their impact on performance.

Fine-tuning allows pretrained features to adapt to the target dataset.

---

## Configuration

```python
DATA_DIR = "path_to_training_dataset"
RESULT_DIR = "path_to_store_results"
```

---

## Output Structure

```bash
Fine-Tuning_Strategies/
│
├── resnet50/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │
│   ├── strategy.txt
│   ├── training_curves.png
│
├── densenet121/
├── efficientnet_b0/
```

---

## Generated Outputs

- Best performing checkpoint
- Strategy logs
- Training and validation accuracy curves

---

# Experiment 3: Few-Shot Learning Analysis

## Objective

Analyze **data efficiency** by training models using only a small subset of the dataset.

Few-shot experiments evaluate how well models generalize with limited training data.

---

## Configuration

```python
DATA_DIR = "path_to_training_dataset"
RESULT_DIR = "path_to_store_results"
```

---

## Output Structure

```bash
FewShot_Results/
│
├── resnet50/
│   ├── checkpoints/
│   ├── experiment.txt
│   ├── loss5.png
│   ├── loss10.png
│   ├── loss20.png
│   ├── accuracy_vs_data.png
│   ├── train_val_gap.png
│
├── densenet121/
├── efficientnet_b0/
```

---

## Generated Outputs

- Accuracy vs training data size
- Loss curves for different data fractions
- Generalization gap analysis
- Model comparison across architectures

---

# Experiment 4: Corruption Robustness Evaluation

## Objective

Evaluate model robustness when the input images are corrupted with noise or distortions.

Robust models should maintain strong performance even under degraded input conditions.

---

## Configuration

```python
DATA_DIR = "path_to_dataset"
MODEL_DIR = "path_to_fine_tuned_weights"
RESULT_DIR = "path_to_store_results"
```

---

## Output Structure

```bash
Robustness_Results/
│
├── resnet50/
│   ├── noise_accuracy.png
│   ├── log.txt
│
├── densenet121/
│   ├── noise_accuracy.png
│   ├── log.txt
│
├── efficientnet_b0/
│   ├── noise_accuracy.png
│   ├── log.txt
```

---

## Generated Outputs

- Accuracy vs corruption severity plots
- Detailed evaluation logs
- Model robustness comparison across architectures

---

# Experiment 5: Layer-Wise Feature Probing

## Objective

Investigate how **semantic representations evolve across network depth**.

Intermediate representations are extracted from:

- Early layers
- Middle layers
- Final layers

Separate linear classifiers are trained on each representation.

---

## Configuration

```python
DATA_DIR = "path_to_dataset"
MODEL_DIR = "fine_tuned_weights_directory"
RESULT_DIR = "representation_analysis_results"
```

---

## Output Structure

```bash
Representation_Analysis/
│
├── resnet50/
│   ├── log.txt
│   ├── accuracy_vs_depth.png
│   ├── pca_early.png
│   ├── pca_middle.png
│   ├── pca_final.png
│
├── densenet121/
├── efficientnet_b0/
```

---

## Generated Outputs

- Validation accuracy vs network depth
- PCA visualizations of features
- Feature norm statistics across layers

These results illustrate the transition from **low level visual features to high level semantic representations**.

---

# Visualization Methods

## PCA

Principal Component Analysis reduces high dimensional feature representations to two dimensions while preserving variance.

## t-SNE

t-SNE reveals local clustering patterns and helps visualize how well different classes separate in feature space.

---

# Performance Metrics

The experiments report several evaluation metrics:

- **Accuracy**
- **Confusion Matrix**
- **Feature Norm Statistics**
- **Training vs Validation Gap**

These metrics help evaluate representation quality, generalization ability, and robustness.

---

# Running the Notebook

Run the notebook sequentially in **Google Colab**.

```bash
1. Mount Google Drive
2. Extract the dataset
3. Run Linear Probe Transfer
4. Train Fine-Tuned Models
5. Run Few-Shot Experiments
6. Evaluate Robustness
7. Perform Layer-Wise Representation Analysis
```

---

# Authors

**Group 19**

- Sarvadnya  
- Ved  
- Siddharth