
# ====================================
# CLINICAL AI RELIABILITY PROJECT
# TRAINING PIPELINE
# ====================================


# ====================================
# IMPORTS
# ====================================

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

# Import objects from data_loader.py
from data_loader import (
    train_loader,
    val_loader,
    test_loader,
    label_mapping,
    device
)

# ====================================
# LOAD PRETRAINED RESNET18
# ====================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# ====================================
# MODIFY FINAL LAYER
# ====================================

num_classes = len(label_mapping)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)


# ====================================
# MOVE MODEL TO DEVICE
# ====================================

model = model.to(device)

print("\nModel initialized successfully! ")

# ====================================
# LOSS FUNCTION
# ====================================

criterion = nn.CrossEntropyLoss()

print("\nLoss function created!")

# ====================================
# OPTIMIZER
# ====================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nOptimizer initialized!")