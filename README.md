# Clinical AI Reliability
## Explainable Deep Learning for Skin Lesion Classification Using ResNet18 and Grad-CAM
---
# Project Overview
This project investigates the reliability and interpretability of deep learning models for clinical skin lesion classification using the HAM10000 dermatology dataset.
The system uses:
- ResNet18 transfer learning
- Fine-tuning strategies
- Data augmentation
- Early stopping
- Learning rate scheduling
- Weighted loss functions
- Grad-CAM explainability
The goal is to improve trustworthy AI in medical imaging by combining strong classification performance with interpretable visual explanations.
---
# Dataset
Dataset used:
HAM10000 ("Human Against Machine with 10000 Training Images")
Dataset contains 7 diagnostic categories:
| Label | Description |
|---|---|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |
Total images:
- 10,015 dermoscopic images
Train/Validation/Test split:
- Train: 70%
- Validation: 15%
- Test: 15%
---
# Model Architecture
Model:
- ResNet18 pretrained on ImageNet
Transfer Learning Strategy:
- Partial fine-tuning
- Unfroze final ResNet layer (`layer4`)
- Replaced final classification layer
Final classifier:
```python
nn.Linear(512, 7)

⸻

Training Features

The pipeline includes:

* Transfer learning
* Partial fine-tuning
* Weighted cross-entropy loss
* Early stopping
* ReduceLROnPlateau scheduler
* Validation checkpointing
* Experiment tracking
* Confusion matrix generation
* Grad-CAM explainability

⸻

Performance

Best Experimental Result

Metric	Value
Test Accuracy	84.30%
Macro F1 Score	0.72
Weighted F1 Score	0.84

⸻

Classification Report

              precision    recall  f1-score
bkl              0.81       0.54      0.65
nv               0.91       0.94      0.92
df               0.60       0.53      0.56
mel              0.64       0.68      0.66
vasc             0.94       0.73      0.82
bcc              0.65       0.83      0.73
akiec            0.66       0.71      0.69

⸻

Training Curves

Loss Curves

Validation Accuracy

⸻

Confusion Matrix

⸻

Explainable AI with Grad-CAM

Grad-CAM visualizations were generated to improve model interpretability and identify image regions contributing most strongly to predictions.

Example Grad-CAM outputs:

Original	Grad-CAM
	

Additional Grad-CAM examples are available in:

plots/gradcam/

⸻

Project Structure

clinical-ai-reliability/
│
├── data/
├── models/
│   └── best_model.pth
│
├── plots/
│   ├── confusion_matrix.png
│   ├── loss_curve.png
│   ├── validation_accuracy.png
│   └── gradcam/
│
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   └── test_metrics.txt
│
├── src/
│   ├── data_loader.py
│   ├── train.py
│   └── gradcam.py
│
├── requirements.txt
├── README.md
└── .gitignore

⸻

Installation

Clone repository:

git clone https://github.com/VenuYerramsetti/clinical-ai-reliability.git

Move into project:

cd clinical-ai-reliability

Create virtual environment:

python -m venv venv

Activate environment:

Mac/Linux:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

⸻

Running Training

python src/train.py

⸻

Running Grad-CAM

python src/gradcam.py

⸻

Research Contributions

This project demonstrates:

* Reliable medical image classification
* Explainable AI for healthcare
* Deep learning interpretability
* Transfer learning optimization
* AI transparency in clinical systems

⸻

Future Improvements

Potential future work includes:

* EfficientNet architectures
* Vision Transformers (ViTs)
* Ensemble learning
* Multi-modal clinical metadata integration
* Uncertainty estimation
* Clinical calibration analysis

⸻

Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* OpenCV

⸻

Author

Venu Yerramsetti

Master’s Applicant — AI/ML Research

Focus Areas:

* Medical AI
* Explainable AI
* Clinical Reliability
* Computer Vision
* Deep Learning

⸻

License

This project is for academic and research purposes.

---
# THEN DO
Save file.
Then run:
```bash
git add .
git commit -m "Added professional project README"
git push

⸻

