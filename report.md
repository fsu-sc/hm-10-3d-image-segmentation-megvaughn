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

# heart mri dataset for training
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

        # Load with SimpleITK
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)

        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label).astype(np.float32)

        # Normalize image to [0, 1]
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-8)

        # Add channel dimension: (1, D, H, W)
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        label_tensor = torch.from_numpy(label_array).unsqueeze(0)

        return image_tensor, label_tensor

# load heart mri dataset
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

# %%

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

```
# Output:




#### Model Evaluation
In your main script (main.py), implement:

- Model training
- Performance evaluation on validation set
- 3D visualization of results
- Calculation of Dice scores and other relevant metrics

```python

```
## Output

