# Algorithms for Image Segmentation based on Deep Learning. Brain Tumor Detection

This repository contains an implementation of ResUNet-based deep learning model for precise segmentation of brain tumors in MRI scans. The model combines the strengths of U-Net and Resnet architectures with a custom loss function to address class imbalance in medical imaging. \
By modifying the base architecture, adjusting the learning rate of the Adam optimisator, and using different training datasets, I was able to train three distinct models.

# Trained models
## Model 1
The first model was trained on a dataset containing only tumor-positive masks. \
I used a learning rate of 0.001 for the Adam optimizer, resulting in a Tversky Index score of 88%.

## Model 2
The second model was also trained tumor-positive masks. \
This time, I used a higher learning rate of 0.05 and also added an epsilon value of 0.1 for the Adam optimizer.
After running the model I got a Tversky Index score of 92%.

## Model 3
The second model was trained on the entire set, which was first split into two subsets: tumor-positive and tumor-negative images. \
Each subset was split into training, validation and testing portions seperately, and then splits were emerged to form a balanced dataset. \
This model also used a learning rate of 0.05 and epsilon value of 0.1 with the Adam optimizer. It achieved a Tversky Index score of 84%.
After running the model I got a Tversky index score of 84%.

## Prerequisites
* Download the dataset from https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation and add it as a kaggle_3m folder
* Python 3.9 or lower
* Required libraries (install with pip install <library_name>)
  * tensorflow
  * keras
  * numpy
  * matplotlib
  * opencv-python
  * scikit-image
  * scikit-learn

You can reuse the same models that I've already trained in test_area_for_models.py


