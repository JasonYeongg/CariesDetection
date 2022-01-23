# Blob Detection
[Automatic Blob Detection for Dental Caries](https://www.researchgate.net/publication/355065789_Automatic_Blob_Detection_for_Dental_Caries "link")

## Result on Caries

| Caries Cover Rate |
|:----------:|
| 80.22% |

![CoverRate](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/Blob%20Detection/README_img/CoverRate.jpg?raw=true "CoverRate")

### SimpleCnn
| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 982 | 10569 | 6017 | 200 | 83.1% | 14% | 24% | 65% |

![SimpleCnn](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/Blob%20Detection/README_img/SimpleCnn.jpg?raw=true "SimpleCnn")

### Blobness
| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 666 | 7359 | 3133 | 83 | 88.9% | 17.5% | 29.2% | 71.4% |

![Blobness](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/Blob%20Detection/README_img/Blobness.jpg?raw=true "Blobness")

## Introduction of each file

> pross.py

先對每一張照片做postprocessing，然後再做blob detection，並把每個detect到的blob存成正方形的照片，並按照是否包含caries（/data/cv裡面的00000和00001）來分類，之後再對這些已經detect完的資料做訓練以及結果偵測

> model/train.py

之後再進行training和validation的資料分割以及用類似Edge frame detection的SimpleCnn來進行訓練（fold）

> predict/predict.py

在這裡用以及在pross進行blob detection後的資料做predict，並同時計算blobness來淘汰不是caries的blob

## Introduction to the paper

### Blob:

![Blob](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/Blob%20Detection/README_img/blob.jpg?raw=true "Blob")

>暗背景上的亮区域，或者亮背景上的暗区域，都可以称为blob。主要利用blob与背景之间的对比度来进行检测。

### Laplacian of Gaussian (LoG):

![Laplacian of Gaussian (LoG)](https://github.com/dentallio/hall-ai/blob/caries_detection/caries_detection/Blob%20Detection/README_img/Laplacian%20of%20Gaussian%20(LoG).jpg?raw=true "Laplacian of Gaussian (LoG)")

>速度最慢，但是最准确的一种算法
>先进行一系列不同尺度的高斯滤波，然后对滤波后的图像做Laplacian运算，将所有的图像进行叠加，局部最大值就是所要检测的blob

### Preprocessing:

120张牙科X光片，对其进行旋转、缩放和调整图像的大小的增强，产生11114张图像,
再對图像还进行gray image scaling和blurring的预处理

### System flow diagram:

![System flow diagram](https://www.researchgate.net/publication/355065789/figure/fig1/AS:1076383186984960@1633641158198/System-flow-diagram_W640.jpg "System flow diagram")

>Feature Extraction:
>Noise Reduction(Gaussian filters)
>Geometric features enhanced(Hessian analysis)

