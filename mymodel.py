# %%
# training implementation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mymodel import UNet3D
from analyze_data import HeartMRIDataset
import os
import numpy as np
from datetime import datetime

# dice loss b/t prediction and actual
def dice_loss(pred, target, smooth=1e-5):
    # [0, 1]
    pred = torch.sigmoid(pred)
    # flatten sample
    pred_flat = pred.view(pred.shape[0], -1)      
    target_flat = target.view(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)

    return 1 - dice.mean()  

# training u net w/ train and valid loops
def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-4):
    
    model = model.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # tensorboard logging
    log_dir = f"runs/heart_mri_unet3d_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # model architecture to tensorboard (try smaller dummy input)
    try:
        dummy_input = torch.randn(1, 1, 32, 64, 64).to(device)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print("Graph logging failed:", e)

    # best validation loss 
    best_val_loss = float("inf")  
    # early stopping
    patience = 10  
    # how long w/out improvement
    trigger = 0  

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        # training phase
        for batch in train_loader:
            # input and mask
            x, y = batch[0].to(device), batch[1].to(device)  

            optimizer.zero_grad()
            # forward pass
            output = model(x)  
            # dice loss
            loss = dice_loss(output, y)  
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # average training loss
        avg_train_loss = np.mean(train_losses)

        # validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                output = model(x)
                loss = dice_loss(output, y)
                val_losses.append(loss.item())

        # average validation loss
        avg_val_loss = np.mean(val_losses)

        # tensorboard curves
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            writer.add_scalar("Memory/Allocated_MB", mem_allocated, epoch)
            writer.add_scalar("Memory/Reserved_MB", mem_reserved, epoch)

        # sample segmentation predictions
        if epoch % 5 == 0:
            pred = torch.sigmoid(output[0]).cpu().detach().numpy()[0]
            label_vis = y[0].cpu().numpy()[0]
            slice_idx = pred.shape[0] // 2  
            writer.add_image("Prediction", pred[slice_idx:slice_idx+1], epoch, dataformats='CHW')
            writer.add_image("Actual", label_vis[slice_idx:slice_idx+1], epoch, dataformats='CHW')

        print(f"Epoch {epoch+1}/{num_epochs} | Train Dice Loss: {avg_train_loss:.4f} | Val Dice Loss: {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger = 0
            torch.save(model.state_dict(), "best_model.pth")  
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    writer.close()  

# main
if __name__ == "__main__":
    # training and validation datasets
    train_dataset = HeartMRIDataset(split='train')
    val_dataset = HeartMRIDataset(split='val')

    # dataloaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # use gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(in_channels=1, out_channels=1)

    # train the model
    train_model(model, train_loader, val_loader, device, num_epochs=100)

# %%
