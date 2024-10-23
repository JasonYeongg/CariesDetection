# nnUNet
[Deep Learning for Caries Detection and Classification](https://www.researchgate.net/publication/354578712_Deep_Learning_for_Caries_Detection_and_Classification "link")

## Result on Caries

| TP  | FN  | Include | Empty (FP) | Sensitivity | Precision | F1  |
|:---:|:---:|:-------:|:----------:|:-----------:|:---------:|:---:|
| 103 | 116 | 104     | 114        | 47%         | 47.5%     | 47% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/nnUNet/Result.jpg?raw=true "Result")

## Introduction of Each File

> nnunetpy/pross.py

Saves URLs as JSON, splits the caries and masks into training and validation datasets, and stores them as JPG files.

> nnunetpy/aug.py

Performs augmentation on the classified training and validation data.

> nnunetpy/2d2unet.py

Converts the images and masks into the format required for nnUNet training.

> nnunetpy/nii22d.py

Converts the predicted data from nnUNet into JPG format.

## Introduction to the nnUNet

> Automatically adapts to any dataset without manual intervention, fully leveraging dataset characteristics to train a basic U-Net model.

![Work Flow](https://miro.medium.com/max/2000/0*PkMBRPa77g-ICW5e.png "Work Flow")

> The workflow is optimized using both data fingerprints (key dataset attributes) and pipeline fingerprints (key design choices of the segmentation algorithm).

> nnU-Net uses heuristic rules to determine hyperparameters related to the data (data fingerprint) to acquire the training data.

> Blueprint parameters (e.g., loss function, optimizer, architecture) and inferred parameters (e.g., image resampling, normalization, batch and patch size) combine with data fingerprints to generate pipeline fingerprints.

> Pipeline fingerprints are then used to train 2D, 3D, and 3D-Cascade U-Net networks. The best average Dice score is determined by evaluating multiple network configurations and post-processing decisions. The best configuration is then applied to predict on the test data.
