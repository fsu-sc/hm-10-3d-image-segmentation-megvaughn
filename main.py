# %%
# model evaluation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D

from mymodel import UNet3D
from analyze_data import HeartMRIDataset

# dice score 
def dice_score(pred, target, threshold=0.5, smooth=1e-5):
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# plot 3D prediction v. actual
def plot_3d_segmentation(pred, label, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    label = (label > threshold).astype(np.uint8)

    fig = plt.figure(figsize=(10, 5))

    # prediction
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.voxels(pred, facecolors='pink', edgecolor='k')
    ax1.set_title("Prediction")

    # actual
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.voxels(label, facecolors='purple', edgecolor='k')
    ax2.set_title("Ground Truth")

    plt.show()

# main evaluation routine
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = UNet3D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # validation dataset
    val_dataset = HeartMRIDataset(split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    dice_scores = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)

            # dice score
            score = dice_score(output, y).item()
            dice_scores.append(score)

            print(f"Sample {i+1} Dice score: {score:.4f}")

            # visualize 1st sample only
            if i == 0:
                pred_np = torch.sigmoid(output[0]).cpu().numpy()[0]
                label_np = y[0].cpu().numpy()[0]
                plot_3d_segmentation(pred_np, label_np)

    print(f"\nAverage Dice score over {len(val_loader)} samples: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    evaluate_model()
