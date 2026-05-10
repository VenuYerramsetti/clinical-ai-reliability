# Clinical AI Reliability: Explainable Skin Lesion Classification Using Transfer Learning and Grad-CAM

## Venu Madhuri Yerramsetti

---

# Abstract

Artificial intelligence systems are increasingly being explored for clinical diagnostic applications; however, reliability and interpretability remain critical challenges in healthcare deployment. This work presents an explainable deep learning framework for multiclass skin lesion classification using the HAM10000 dermatology dataset. A pretrained ResNet18 architecture was optimized using transfer learning, partial fine-tuning, weighted cross-entropy loss, and learning rate scheduling to improve performance on imbalanced clinical data. Grad-CAM visualizations were integrated to provide interpretable heatmaps highlighting image regions contributing most strongly to model predictions.

The proposed system achieved a test accuracy of 84.30% with strong weighted F1 performance across seven lesion categories. Experimental results demonstrate that explainability methods can improve transparency while maintaining competitive predictive performance. This project contributes toward the development of trustworthy clinical AI systems for healthcare applications.

---

# 1. Introduction

Artificial intelligence has demonstrated significant potential in medical image analysis, particularly for disease detection and diagnostic assistance. Deep learning methods have achieved high performance across multiple healthcare tasks, including radiology, pathology, dermatology, and ophthalmology. Despite strong predictive capabilities, concerns regarding model interpretability and clinical trustworthiness remain major barriers to deployment in real-world healthcare systems.

Skin cancer is among the most common forms of cancer worldwide, and early diagnosis significantly improves treatment outcomes. Dermoscopic image analysis using deep learning has emerged as a promising approach for automated skin lesion classification. However, many clinical AI systems behave as black-box models, limiting physician trust and reducing transparency in diagnostic reasoning.

This work focuses on reliable and explainable skin lesion classification using deep learning. The project investigates transfer learning with ResNet18 combined with Grad-CAM explainability to improve both predictive performance and interpretability. The objective is to create a clinically relevant AI pipeline capable of supporting trustworthy medical decision systems.

The main contributions of this work include:

- Development of a deep learning pipeline for multiclass skin lesion classification
- Optimization using partial fine-tuning and weighted loss functions
- Integration of Grad-CAM for explainable AI visualization
- Experimental evaluation on the HAM10000 dataset
- Analysis of interpretability and reliability in clinical AI systems

---

# 2. Related Work

Deep learning has become increasingly prominent in medical image classification tasks. Convolutional Neural Networks (CNNs) such as ResNet, DenseNet, EfficientNet, and Vision Transformers have demonstrated strong performance in dermatological image analysis.

Transfer learning has proven especially effective in medical imaging due to limited labeled clinical datasets. Pretrained ImageNet models enable feature reuse while reducing computational requirements and training time.

Recent research has also emphasized explainable AI (XAI) techniques for healthcare systems. Grad-CAM is one of the most widely adopted visualization approaches for interpreting CNN predictions by highlighting image regions contributing to classification decisions.

Despite advances in performance, many studies still prioritize accuracy over interpretability and reliability. This work aims to bridge that gap by combining strong classification performance with explainability-focused evaluation.

---

# 3. Dataset

## 3.1 HAM10000 Dataset

The HAM10000 dataset contains 10,015 dermoscopic images categorized into seven diagnostic lesion classes:

| Label | Description |
|---|---|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |

The dataset presents substantial class imbalance, making reliable multiclass classification challenging.

---

## 3.2 Data Preprocessing

Images were resized to 224 × 224 resolution and converted into tensor representations for deep learning processing. Data augmentation and normalization techniques were applied during training to improve generalization performance.

The dataset was divided into:

| Split | Percentage |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Stratified sampling was used to preserve class distributions across splits.

---

# 4. Methodology

## 4.1 Transfer Learning

A pretrained ResNet18 architecture initialized with ImageNet weights was used as the base model. Transfer learning was selected due to limited clinical dataset size and strong feature extraction capabilities of pretrained CNNs.

---

## 4.2 Partial Fine-Tuning

Instead of fully retraining the entire network, the early layers of ResNet18 were frozen while the final convolutional block (`layer4`) and classification head were fine-tuned.

This strategy improved:

- training stability,
- generalization,
- computational efficiency,
- minority class learning.

The final fully connected layer was replaced with:

```python
nn.Linear(512, 7)
```

---

## 4.3 Loss Function

Weighted Cross-Entropy Loss was used to mitigate class imbalance. Class weights were computed inversely proportional to class frequencies to improve minority class representation.

---

## 4.4 Optimization

The model was trained using the Adam optimizer with learning rate scheduling via ReduceLROnPlateau. Early stopping was implemented to reduce overfitting and improve validation stability.

---

## 4.5 Explainable AI with Grad-CAM

Grad-CAM visualizations were generated to improve model interpretability. Heatmaps highlight image regions contributing most strongly to model predictions, enabling visual inspection of clinically relevant attention regions.

---

# 5. Experimental Setup

## 5.1 Hardware

Experiments were conducted using Apple Silicon acceleration with PyTorch MPS backend support.

---

## 5.2 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 15 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Enabled |

---

# 6. Results

## 6.1 Overall Performance

| Metric | Value |
|---|---|
| Test Accuracy | 84.30% |
| Weighted F1 Score | 0.84 |
| Macro F1 Score | 0.72 |

---

## 6.2 Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| bkl | 0.81 | 0.54 | 0.65 |
| nv | 0.91 | 0.94 | 0.92 |
| df | 0.60 | 0.53 | 0.56 |
| mel | 0.64 | 0.68 | 0.66 |
| vasc | 0.94 | 0.73 | 0.82 |
| bcc | 0.65 | 0.83 | 0.73 |
| akiec | 0.66 | 0.71 | 0.69 |

---

## 6.3 Confusion Matrix

The confusion matrix demonstrates strong classification performance for majority classes while indicating remaining challenges in minority class differentiation.

### Figure to Include
- `plots/confusion_matrix.png`

---

## 6.4 Training Curves

Training and validation curves indicate stable convergence behavior with reduced overfitting due to early stopping and scheduler-based optimization.

### Figures to Include
- `plots/loss_curve.png`
- `plots/validation_accuracy.png`

---

# 7. Explainability Analysis

Grad-CAM visualizations demonstrate that the model primarily focuses on lesion boundaries, pigmentation regions, and clinically relevant structures when generating predictions.

The explainability results suggest that the network is learning medically meaningful visual patterns rather than relying on irrelevant background features.

### Figures to Include
- `plots/gradcam/sample_1.png`
- `plots/gradcam/sample_2.png`
- `plots/gradcam/sample_3.png`
- `plots/gradcam/sample_4.png`
- `plots/gradcam/sample_5.png`

---

# 8. Limitations

Several limitations remain in the current system:

- Class imbalance affects minority category performance
- Limited dataset diversity may impact generalization
- Single-model architecture without ensemble methods
- Lack of uncertainty estimation for clinical deployment
- Limited metadata integration

Future work should investigate calibration analysis, uncertainty-aware prediction, multimodal learning, and larger clinical datasets.

---

# 9. Future Work

Potential future directions include:

- EfficientNet and Vision Transformer architectures
- Ensemble deep learning methods
- Federated medical AI systems
- Uncertainty estimation and calibration
- Multimodal metadata integration
- Clinical decision support systems
- Human-AI collaborative diagnosis

---

# 10. Conclusion

This work presented an explainable deep learning framework for reliable skin lesion classification using transfer learning and Grad-CAM. Experimental results demonstrate that combining partial fine-tuning, weighted optimization, and explainability techniques can produce strong multiclass classification performance while improving model interpretability.

The project contributes toward trustworthy medical AI research by emphasizing transparency and clinical reliability alongside predictive accuracy. Future work will focus on uncertainty-aware clinical AI systems and advanced explainability approaches for healthcare deployment.

---

# References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 Dataset.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
4. Esteva, A., et al. (2017). Dermatologist-level classification of skin cancer with deep neural networks.