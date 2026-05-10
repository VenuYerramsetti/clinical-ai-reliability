
# ====================================
# CLINICAL AI RELIABILITY PROJECT
# GRAD-CAM VISUALIZATION
# ====================================

# ====================================
# IMPORTS
# ====================================

import os
import random
import numpy as np

import cv2

import torch
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

import matplotlib.pyplot as plt

from PIL import Image

# ====================================
# IMPORT DATA
# ====================================

from data_loader import (
    test_dataset,
    label_mapping,
    device
)

# ====================================
# CREATE DIRECTORY
# ====================================

os.makedirs(
    "plots/gradcam",
    exist_ok=True
)

# ====================================
# LOAD MODEL
# ====================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

num_classes = len(label_mapping)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nModel loaded successfully!")

# ====================================
# TARGET LAYER
# ====================================

target_layer = model.layer4[-1]

# ====================================
# HOOK STORAGE
# ====================================

activations = []
gradients = []

# ====================================
# FORWARD HOOK
# ====================================

def forward_hook(module, input, output):

    activations.append(output)

# ====================================
# BACKWARD HOOK
# ====================================

def backward_hook(module, grad_input, grad_output):

    gradients.append(grad_output[0])

# ====================================
# REGISTER HOOKS
# ====================================

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)

# ====================================
# LABEL NAMES
# ====================================

idx_to_class = {
    v: k for k, v in label_mapping.items()
}

# ====================================
# SELECT RANDOM TEST SAMPLES
# ====================================

random.seed(42)

sample_indices = random.sample(
    range(len(test_dataset)),
    5
)

print("\nGenerating Grad-CAMs...")

# ====================================
# GENERATE GRAD-CAM
# ====================================

for sample_num, idx in enumerate(sample_indices):

    # ====================================
    # LOAD IMAGE
    # ====================================

    image_tensor, label = test_dataset[idx]

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    # ====================================
    # CLEAR STORAGE
    # ====================================

    activations.clear()
    gradients.clear()

    # ====================================
    # FORWARD PASS
    # ====================================

    output = model(input_tensor)

    predicted_class = output.argmax(dim=1).item()

    # ====================================
    # BACKWARD PASS
    # ====================================

    model.zero_grad()

    class_score = output[0, predicted_class]

    class_score.backward()

    # ====================================
    # GET ACTIVATIONS & GRADIENTS
    # ====================================

    activation = activations[0]

    gradient = gradients[0]

    # ====================================
    # GLOBAL AVERAGE POOLING
    # ====================================

    pooled_gradients = torch.mean(
        gradient,
        dim=[0, 2, 3]
    )

    # ====================================
    # WEIGHT ACTIVATION MAPS
    # ====================================

    for i in range(
        activation.shape[1]
    ):

        activation[:, i, :, :] *= (
            pooled_gradients[i]
        )

    # ====================================
    # CREATE HEATMAP
    # ====================================

    heatmap = torch.mean(
        activation,
        dim=1
    ).squeeze()

    heatmap = F.relu(heatmap)

    heatmap /= torch.max(heatmap)

    heatmap = (
        heatmap
        .detach()
        .cpu()
        .numpy()
    )

    # ====================================
    # ORIGINAL IMAGE
    # ====================================

    image_np = (
        image_tensor
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    image_np = np.clip(
        image_np,
        0,
        1
    )

    # ====================================
    # RESIZE HEATMAP
    # ====================================

    heatmap = cv2.resize(
        heatmap,
        (
            image_np.shape[1],
            image_np.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # ====================================
    # OVERLAY
    # ====================================

    superimposed_img = (
        heatmap * 0.4 +
        np.uint8(image_np * 255)
    )

    superimposed_img = np.clip(
        superimposed_img,
        0,
        255
    ).astype(np.uint8)

    # ====================================
    # SAVE FIGURE
    # ====================================

    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)

    plt.imshow(image_np)

    plt.title(
        f"True: "
        f"{idx_to_class[label]}"
    )

    plt.axis("off")

    # Grad-CAM
    plt.subplot(1, 2, 2)

    plt.imshow(
        cv2.cvtColor(
            superimposed_img,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Predicted: "
        f"{idx_to_class[predicted_class]}"
    )

    plt.axis("off")

    plt.tight_layout()

    save_path = (
        f"plots/gradcam/"
        f"sample_{sample_num + 1}.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(
        f"Saved: {save_path}"
    )

print("\nGrad-CAM generation completed!")