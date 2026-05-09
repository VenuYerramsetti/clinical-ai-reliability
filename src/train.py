# ====================================
# CLINICAL AI RELIABILITY PROJECT
# TRAINING PIPELINE
# ====================================


# ====================================
# IMPORTS
# ====================================

import time

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

print("\nModel initialized successfully!")


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


# ====================================
# TRAINING CONFIG
# ====================================

NUM_EPOCHS = 3

print("\nTraining configuration ready!")


# ====================================
# START TRAINING TIMER
# ====================================

training_start_time = time.time()


# ====================================
# TRAINING LOOP
# ====================================

for epoch in range(NUM_EPOCHS):

    # Start epoch timer
    epoch_start_time = time.time()

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # Set model to training mode
    model.train()

    running_loss = 0.0

    # Loop through batches
    for images, labels in train_loader:

        # Move tensors to GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Track total loss
        running_loss += loss.item()

    # Average epoch loss
    epoch_loss = running_loss / len(train_loader)

    print(f"Training Loss: {epoch_loss:.4f}")

    # End epoch timer
    epoch_end_time = time.time()

    epoch_duration = (
        epoch_end_time - epoch_start_time
    )

    print(
        f"Epoch Time: "
        f"{epoch_duration:.2f} seconds"
    )


# ====================================
# TOTAL TRAINING TIME
# ====================================

training_end_time = time.time()

total_training_time = (
    training_end_time - training_start_time
)

print("\nTraining completed!")

print(
    f"Total Training Time: "
    f"{total_training_time:.2f} seconds"
)