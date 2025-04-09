# Homework 10 Report: Heart MRI Segmentation using Deep Learning

## Personal Details
**Name:** Megan Vaughn
**Date:** April 9, 2025
**Course:** ISC 5935
**Instructor:** Olmo Zavala Romero

## Homework Questions and Answers

#### Dataset Exploration
Create a file called analyze_data.py that:

- Loads the Heart MRI dataset from the NIfTI files
- Displays basic statistics (number of images, image dimensions, voxel spacing)
- Visualizes sample slices from different orientations (axial, sagittal, coronal)
- (Optional) Shows the distribution of segmentation volumes

```python
# %%
# data exploration

import os
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# heart mri dataset for training.py
class HeartMRIDataset(Dataset):
    def __init__(self, root_dir="/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart", split='train'):
        self.image_dir = os.path.join(root_dir, "imagesTr")
        self.label_dir = os.path.join(root_dir, "labelsTr")

        all_files = sorted(os.listdir(self.image_dir))
        split_idx = int(0.8 * len(all_files))

        if split == 'train':
            self.file_list = all_files[:split_idx]
        else:
            self.file_list = all_files[split_idx:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.file_list[idx])
        label_path = os.path.join(self.label_dir, self.file_list[idx].replace("_0000", ""))

        # load image and label using SimpleITK
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label).astype(np.float32)

        # normalize to [0, 1]
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)

        # crop size (D, H, W)
        crop_d, crop_h, crop_w = 64, 128, 128
        D, H, W = image_array.shape

        # make sure crop fits
        assert D >= crop_d and H >= crop_h and W >= crop_w, "Crop size is larger than input volume."

        # random crop start indices
        start_d = np.random.randint(0, D - crop_d + 1)
        start_h = np.random.randint(0, H - crop_h + 1)
        start_w = np.random.randint(0, W - crop_w + 1)

        # apply crop to both image and label
        image_array = image_array[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]
        label_array = label_array[start_d:start_d+crop_d, start_h:start_h+crop_h, start_w:start_w+crop_w]

        # convert to tensors and add channel dimension: (1, D, H, W)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # shape (1, D, H, W)
        label_tensor = torch.from_numpy(label_array).unsqueeze(0)

        return image_tensor, label_tensor


# load heart mri dataset for analyze_data.py
if __name__ == "__main__":
    # load heart mri dataset
    data_dir = "/home/osz09/DATA_SharedClasses/SharedDatasets/MedicalDecathlon/Task02_Heart"
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    # image and label file lists
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])

    # number of images and labels
    print(f"Number of training images: {len(image_files)}")
    print(f"Number of training labels: {len(label_files)}")

    # sample image and label
    sample_image_path = os.path.join(images_dir, image_files[0])
    sample_label_path = os.path.join(labels_dir, label_files[0])

    # read images using SimpleITK
    img = sitk.ReadImage(sample_image_path)
    label = sitk.ReadImage(sample_label_path)

    # numpy arrays [slices, height, width]
    img_array = sitk.GetArrayFromImage(img)
    label_array = sitk.GetArrayFromImage(label)

    # stats
    print(f"Sample image shape (slices, height, width): {img_array.shape}")
    print(f"Voxel spacing (z, y, x): {img.GetSpacing()}")

    # center slice indices
    z_center = img_array.shape[0] // 2
    y_center = img_array.shape[1] // 2
    x_center = img_array.shape[2] // 2

    # plot axial, sagittal, coronal views
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0, 0].imshow(img_array[z_center, :, :], cmap='gray')
    axes[0, 0].set_title("MRI - Axial")

    axes[0, 1].imshow(img_array[:, :, x_center], cmap='gray')
    axes[0, 1].set_title("MRI - Sagittal")

    axes[0, 2].imshow(img_array[:, y_center, :], cmap='gray')
    axes[0, 2].set_title("MRI - Coronal")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # segmentation volume distribution
    volumes = []
    for f in label_files:
        label_path = os.path.join(labels_dir, f)
        label_img = sitk.ReadImage(label_path)
        label_array = sitk.GetArrayFromImage(label_img)
        volume = np.sum(label_array > 0)
        volumes.append(volume)

    print("Volume stats:")
    print(f"  Mean volume: {np.mean(volumes):.2f} voxels")
    print(f"  Min volume: {np.min(volumes)} voxels")
    print(f"  Max volume: {np.max(volumes)} voxels")

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(volumes, bins=10, color='slateblue', edgecolor='pink', alpha=0.8)
    plt.title("Distribution of Segmentation Volumes")
    plt.xlabel("Volume (number of voxels labeled > 0)")
    plt.ylabel("Number of cases")
    plt.show()


```

# Output:

Number of training images: 20

Number of training labels: 20

Sample image shape (slices, height, width): (130, 320, 320)

Voxel spacing (z, y, x): (1.25, 1.25, 1.3700000047683716)

![data_exploration_axial_sag_cor](https://github.com/user-attachments/assets/bedd7fa5-97aa-4b1b-a30e-3668210a4a38)

Volume stats:

  Mean volume: 46750.75 voxels
  
  Min volume: 32040 voxels
  
  Max volume: 68148 voxels

![data_exploration_histo](https://github.com/user-attachments/assets/2deea097-35af-4e73-b9be-7665894fec92)


#### Model Architecture 
Design and implement a 3D CNN in mymodel.py that includes:

- 3D convolutional layers with appropriate kernel sizes
- 3D batch normalization layers
- 3D max pooling layers
- 3D upsampling / transposed convolutional or other upsampling layers
- Skip connections (recommended for U-Net style architectures)
  
Your model description should include:

- Summary of the model architecture (in your own words)
- Number of parameters
- Visual representation of the model using tensorboard

```python
# %%
# model architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

# 3d cnn, conv layers, batch norm layers, max pooling layers, upsampling layers, 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 2 stacked 3d conv layers
        super(DoubleConv, self).__init__()
        # sequential block
        self.double_conv = nn.Sequential(
            # conv layer
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # batch norm layer
            nn.BatchNorm3d(out_channels),
            # relu layer
            nn.ReLU(inplace=True),
            # second conv layer
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    # forward pass
    def forward(self, x):
        return self.double_conv(x)

# u net model for segmentation
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()

        # encoder (downsamp)
        self.encoder_layers = nn.ModuleList()  
        # max pooling layer (downsamp)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  

        # encoder path (contracting)
        for feature in features:
            self.encoder_layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # bottleneck, deepest layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # decoder (upsamp)
        # transposed conv layer
        self.upconvs = nn.ModuleList()       
        self.decoder_layers = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)  # Upsample
            )
            self.decoder_layers.append(DoubleConv(feature * 2, feature))  # Combine with skip connection

        # final conv layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # forward pass
    def forward(self, x):
        skip_connections = []  # To store outputs from encoder for skip connections

        # encoder (downsamp)
        for encoder in self.encoder_layers:
            x = encoder(x)
            # skip connections
            skip_connections.append(x)  
            x = self.pool(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder (upsamp)
        skip_connections = skip_connections[::-1]

        # decoder (upsamp, concat skip) 
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)  # Upsample
            skip = skip_connections[i]

            # match skip size
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                              0, skip.shape[-2] - x.shape[-2],
                              0, skip.shape[-3] - x.shape[-3]])

            # concat skip connection
            x = torch.cat((skip, x), dim=1)  
            x = self.decoder_layers[i](x)  

        return self.final_conv(x)

# number of params
model = UNet3D(in_channels=1, out_channels=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")

# tensorboard
from torch.utils.tensorboard import SummaryWriter

# model
model = UNet3D(in_channels=1, out_channels=1)

# dummy input (batch size 1, 1 channel, depth=64, height=128, width=128)
dummy_input = torch.randn(1, 1, 64, 128, 128)

# tensorboard writer
writer = SummaryWriter("runs/unet3d_graph")

# model graph
writer.add_graph(model, dummy_input)
writer.close()
#tensorboard --logdir=runs

# %%
```
# Output:
Total trainable parameters: 22,581,217

![mymodel_tensorboard](https://github.com/user-attachments/assets/3f782423-f32a-4c0a-bb61-a8cc09550143)



#### Training Implementation
Create a training.py script that implements:

- Training loop with batches and epochs
- Validation after each epoch
- Dice loss calculation (see Section 3.1 for details)
- Early stopping (optional)
- Learning rate scheduling (optional)
  
TensorBoard logging should include:

- Training and validation Dice loss curves
- Model computational graph
- Example segmentation predictions (optional)
- Memory usage statistics (optional)


```python
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
    print("TensorBoard logs will be written to:", log_dir)


    # model architecture to tensorboard
    sample_input = next(iter(train_loader))[0].to(device)  # one batch
    writer.add_graph(model, sample_input)

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

        # tensorboard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

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
```
# Output:

Epoch 1/100 | Train Dice Loss: 0.9463 | Val Dice Loss: 0.9725
Epoch 2/100 | Train Dice Loss: 0.9552 | Val Dice Loss: 0.9375
Epoch 3/100 | Train Dice Loss: 0.9460 | Val Dice Loss: 0.9757
Epoch 4/100 | Train Dice Loss: 0.9485 | Val Dice Loss: 0.9254
Epoch 5/100 | Train Dice Loss: 0.9129 | Val Dice Loss: 0.9827
Epoch 6/100 | Train Dice Loss: 0.9497 | Val Dice Loss: 0.9817
Epoch 7/100 | Train Dice Loss: 0.9157 | Val Dice Loss: 0.8979
Epoch 8/100 | Train Dice Loss: 0.9294 | Val Dice Loss: 0.9057
Epoch 9/100 | Train Dice Loss: 0.9212 | Val Dice Loss: 0.9160
Epoch 10/100 | Train Dice Loss: 0.9268 | Val Dice Loss: 0.9770
Epoch 11/100 | Train Dice Loss: 0.9333 | Val Dice Loss: 0.9248
Epoch 12/100 | Train Dice Loss: 0.9022 | Val Dice Loss: 0.9457
Epoch 13/100 | Train Dice Loss: 0.9189 | Val Dice Loss: 0.9567
Epoch 14/100 | Train Dice Loss: 0.9595 | Val Dice Loss: 0.9190
Epoch 15/100 | Train Dice Loss: 0.9347 | Val Dice Loss: 0.9309
Epoch 16/100 | Train Dice Loss: 0.8999 | Val Dice Loss: 0.9401
Epoch 17/100 | Train Dice Loss: 0.9371 | Val Dice Loss: 0.9690
Early stopping triggered.

Training Dice Loss:

<img width="369" alt="training_train_loss" src="https://github.com/user-attachments/assets/542f8e78-d239-4488-a9a8-0998c0c9a176" />

Validation Dice Loss:

<img width="360" alt="training_val_loss" src="https://github.com/user-attachments/assets/c39aaf87-b188-4c05-98ed-1a41832d1dec" />

Model Computational Graph:
![training_comp_graph](https://github.com/user-attachments/assets/fe8ae60b-d5c6-4f60-8af4-18de673f08e5)

Example Segmentation Predictions: 

<img width="389" alt="training_pred_red" src="https://github.com/user-attachments/assets/efde238b-7478-4c40-b4fd-c011075606aa" />

<img width="385" alt="training_actual_red" src="https://github.com/user-attachments/assets/5c296385-84a7-4d93-9976-71a863e0e5ea" />

<img width="387" alt="training_pred_blue" src="https://github.com/user-attachments/assets/929c3f66-b7f4-4f82-8d99-db0bc994ad35" />

<img width="380" alt="training_actual_blue" src="https://github.com/user-attachments/assets/74d832ce-2227-4b6c-acc8-9f08def68be5" />

Memory Usage Statistics:

<img width="368" alt="training_allocated" src="https://github.com/user-attachments/assets/69822ef3-ee9f-4748-9bd8-a45976347b17" />

<img width="355" alt="training_reserved" src="https://github.com/user-attachments/assets/746555fc-8613-44d2-82d7-68a8f8fd698f" />

Learning Rate Graph:

<img width="359" alt="training_learning_rate" src="https://github.com/user-attachments/assets/0331a468-76b2-4fc0-b36d-a686f26938c1" />


#### Model Evaluation
In your main script (main.py), implement:

- Model training
- Performance evaluation on validation set
- 3D visualization of results
- Calculation of Dice scores and other relevant metrics

```python

```
## Output

