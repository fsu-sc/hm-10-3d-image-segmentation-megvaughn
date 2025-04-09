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
