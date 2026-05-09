# ====================================
# CLINICAL AI RELIABILITY PROJECT
# TRAINING PIPELINE
# ====================================


# ====================================
# IMPORTS
# ====================================

import time
import numpy as np

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

best_val_accuracy = 0.0

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
# VALIDATION LOOP
# ====================================

model.eval()

val_running_loss = 0.0

correct_predictions = 0
total_predictions = 0

# Disable gradient calculations
with torch.no_grad():

    for images, labels in val_loader:

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Validation loss
        loss = criterion(outputs, labels)

        val_running_loss += loss.item()

        # Predicted class
        _, predicted = torch.max(outputs, 1)

        # Count correct predictions
        correct_predictions += (
            predicted == labels
        ).sum().item()

        # Total predictions
        total_predictions += labels.size(0)


# Average validation loss
val_loss = (
    val_running_loss / len(val_loader)
)

# Validation accuracy
val_accuracy = (
    correct_predictions / total_predictions
) * 100


print(f"Validation Loss: {val_loss:.4f}")

print(
    f"Validation Accuracy: "
    f"{val_accuracy:.2f}%"
)

# ====================================
# SAVE BEST MODEL
# ====================================

if val_accuracy > best_val_accuracy:

    best_val_accuracy = val_accuracy

    torch.save(
        model.state_dict(),
        "best_model.pth"
    )

    print(
        "\nBest model saved!"
    )

# ====================================
# SAVE TO MODELS FOLDER 
# ====================================   

torch.save(
    model.state_dict(),
    "models/best_model.pth"
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