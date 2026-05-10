# ====================================
# CLINICAL AI RELIABILITY PROJECT
# COMPLETE TRAINING PIPELINE
# ====================================

# ====================================
# IMPORTS
# ====================================

import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

# ====================================
# IMPORT DATA LOADERS
# ====================================

from data_loader import (
    train_loader,
    val_loader,
    test_loader,
    label_mapping,
    device
)

# ====================================
# CREATE DIRECTORIES
# ====================================

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ====================================
# RANDOM SEED
# ====================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ====================================
# LOAD PRETRAINED RESNET18
# ====================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

print("\nModel loaded successfully!")

# ====================================
# PARTIAL FINE-TUNING
# ====================================

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# ====================================
# MODIFY FINAL LAYER
# ====================================

num_classes = len(label_mapping)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

# Train classifier head
for param in model.fc.parameters():
    param.requires_grad = True

print("\nFinal layer:")
print(model.fc)

# ====================================
# MOVE MODEL TO DEVICE
# ====================================

model = model.to(device)

print("\nModel initialized successfully!")

# ====================================
# CLASS WEIGHTS
# ====================================

class_counts = torch.tensor([
    1099,
    6705,
    115,
    1113,
    142,
    514,
    327
], dtype=torch.float32)

class_weights = (
    class_counts.sum() / class_counts
)

class_weights = (
    class_weights / class_weights.sum()
)

class_weights = class_weights.to(device)

print("\nClass weights:")
print(class_weights)

# ====================================
# LOSS FUNCTION
# ====================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("\nLoss function created!")

# ====================================
# OPTIMIZER
# ====================================

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)

print("\nOptimizer initialized!")

# ====================================
# LR SCHEDULER
# ====================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.5
)

# ====================================
# TRAINING CONFIG
# ====================================

NUM_EPOCHS = 15

best_val_loss = float("inf")

patience = 5
patience_counter = 0

print("\nTraining configuration ready!")

# ====================================
# METRIC TRACKING
# ====================================

train_losses = []
val_losses = []
val_accuracies = []

# ====================================
# TRAINING TIMER
# ====================================

training_start_time = time.time()

# ====================================
# TRAINING LOOP
# ====================================

for epoch in range(NUM_EPOCHS):

    epoch_start_time = time.time()

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # ====================================
    # TRAINING
    # ====================================

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = (
        running_loss / len(train_loader)
    )

    train_losses.append(epoch_train_loss)

    print(
        f"Training Loss: "
        f"{epoch_train_loss:.4f}"
    )

    # ====================================
    # VALIDATION
    # ====================================

    model.eval()

    val_running_loss = 0.0

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            correct_predictions += (
                predicted == labels
            ).sum().item()

            total_predictions += labels.size(0)

    epoch_val_loss = (
        val_running_loss / len(val_loader)
    )

    val_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    val_losses.append(epoch_val_loss)

    val_accuracies.append(val_accuracy)

    print(
        f"Validation Loss: "
        f"{epoch_val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    # ====================================
    # LEARNING RATE
    # ====================================

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate: {current_lr}")

    # ====================================
    # LR SCHEDULER
    # ====================================

    scheduler.step(epoch_val_loss)

    # ====================================
    # SAVE BEST MODEL
    # ====================================

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print("\nBest model saved!")

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/{patience})"
        )

    # ====================================
    # EARLY STOPPING
    # ====================================

    if patience_counter >= patience:

        print("\nEarly stopping triggered!")

        break

    # ====================================
    # EPOCH TIME
    # ====================================

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

# ====================================
# LOAD BEST MODEL
# ====================================

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

print("\nBest model loaded!")

# ====================================
# TEST EVALUATION
# ====================================

model.eval()

test_loss = 0.0

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

# ====================================
# TEST METRICS
# ====================================

test_loss = test_loss / len(test_loader)

test_accuracy = (
    100 * correct / total
)

print("\nTest Results")

print(f"Test Loss: {test_loss:.4f}")

print(f"Test Accuracy: {test_accuracy:.2f}%")

# ====================================
# CLASSIFICATION REPORT
# ====================================

class_names = list(
    label_mapping.keys()
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:\n")

print(report)

# ====================================
# SAVE REPORTS
# ====================================

with open(
    "reports/classification_report.txt",
    "w"
) as f:

    f.write(report)

with open(
    "reports/test_metrics.txt",
    "w"
) as f:

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%\n"
    )

    f.write(
        f"Test Loss: "
        f"{test_loss:.4f}\n"
    )

    f.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.4f}\n"
    )

    f.write(
        f"Training Time: "
        f"{total_training_time:.2f} seconds\n"
    )

# ====================================
# CONFUSION MATRIX
# ====================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

np.savetxt(
    "reports/confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

# ====================================
# PLOT CONFUSION MATRIX
# ====================================

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    ticks=np.arange(len(class_names)),
    labels=class_names,
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(class_names)),
    labels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.title("Confusion Matrix")

for i in range(len(class_names)):
    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "plots/confusion_matrix.png"
)

plt.close()

# ====================================
# LOSS CURVE
# ====================================

plt.figure(figsize=(10, 6))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Loss Curves")

plt.legend()

plt.grid(True)

plt.savefig(
    "plots/loss_curve.png"
)

plt.close()

# ====================================
# VALIDATION ACCURACY CURVE
# ====================================

plt.figure(figsize=(10, 6))

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "plots/validation_accuracy.png"
)

plt.close()

print("\nPlots saved successfully!")

print("\nSaved files:")

print("- models/best_model.pth")
print("- reports/classification_report.txt")
print("- reports/test_metrics.txt")
print("- reports/confusion_matrix.csv")
print("- plots/confusion_matrix.png")
print("- plots/loss_curve.png")
print("- plots/validation_accuracy.png")