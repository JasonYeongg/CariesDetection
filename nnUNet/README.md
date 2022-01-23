# nnUNet
[Deep Learning for Caries Detection and Classification](https://www.researchgate.net/publication/354578712_Deep_Learning_for_Caries_Detection_and_Classification "link")

## Result on Caries

| TP | FN | include | empty(FP) | Sensitivity | Precision | F1 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 103 | 116 | 104 | 114 | 47% | 47.5% | 47% |

![Result](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/nnUNet/Result.jpg?raw=true "Result")


## Introduction of each file

> nnunetpy/pross.py

把url存成json後再把裡面的caries和mask，再分成training和validation資料並存成jpg

> nnunetpy/aug.py

對分類好的training和validation的資料做augmentation

> nnunetpy/2d2unet.py

將照片和mask轉換成提供nnunet訓練的格式

> nnunetpy/nii22d.py

將nnUNet預測完的資料轉換成jpg

## Introduction to the nnUNet

>可以自动适应任意数据集,且无需人工介入，充分利用数据集的特点训练基本的U-Net模型。

![Work Flow](https://miro.medium.com/max/2000/0*PkMBRPa77g-ICW5e.png "Work Flow")

>根据data fingerprint(数据集的关键属性)和pipeline fingerprint(分割算法的关键设计选择)来制定流水线优化问题

>nnU-Net使用heuristic rule来确定与数据相关的hyper-parameters(data fingerprint)，以獲取训练数据。

>blueprint parameters ( loss function, optimizer,architecture)和inferred parameters ( image resampling, normalization, batch and patch size) 与data fingerprint一起产生pipeline fingerprints。

>pipeline fingerprints則使用迄今为止确定的hyper-parameters为2D, 3D and 3D-Cascade U-Net進行network training。不同网络配置的集合，以及postprocessing决定了训练数据的最佳平均Dice系数。然后，最佳配置将被用于产生对test data的预测。


