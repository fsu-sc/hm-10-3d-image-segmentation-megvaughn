# Homework 10 Report:Heart MRI Segmentation using Deep Learning

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

```

# Output:





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

```
# Output:




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

