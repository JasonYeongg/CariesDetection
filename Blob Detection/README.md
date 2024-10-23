# Blob Detection
[Automatic Blob Detection for Dental Caries](https://www.researchgate.net/publication/355065789_Automatic_Blob_Detection_for_Dental_Caries "link")

## Result on Caries

| Caries Cover Rate |
|:----------:|
| 80.22% |

![CoverRate](https://github.com/jasonyeong/CariesDetection/blob/master/Blob%20Detection/README_img/CoverRate.jpg?raw=true "CoverRate")

### SimpleCnn
| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 982 | 10569 | 6017 | 200 | 83.1% | 14% | 24% | 65% |

![SimpleCnn](https://github.com/jasonyeong/CariesDetection/blob/master/Blob%20Detection/README_img/SimpleCnn.jpg?raw=true "SimpleCnn")

### Blobness
| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 666 | 7359 | 3133 | 83 | 88.9% | 17.5% | 29.2% | 71.4% |

![Blobness](https://github.com/jasonyeong/CariesDetection/blob/master/Blob%20Detection/README_img/Blobness.jpg?raw=true "Blobness")

## Introduction of each file

> pross.py

First, postprocessing is applied to each image, followed by blob detection. Each detected blob is saved as a square image and classified based on whether it contains caries (according to 00000 and 00001 in /data/cv). After that, the detected data is used for training and result detection.

> model/train.py

Then, the data is split for training and validation, using a SimpleCnn model similar to Edge frame detection for training (fold).

> predict/predict.py

Here, the data from blob detection in the pross stage is used to make predictions, while also calculating "blobness" to eliminate blobs that are not caries.

## Introduction to the paper

### Blob:

![Blob](https://github.com/jasonyeong/CariesDetection/blob/master/Blob%20Detection/README_img/blob.jpg?raw=true "Blob")

>Bright areas on a dark background, or dark areas on a bright background, are referred to as blobs. Detection is mainly based on the contrast between the blob and the background.

### Laplacian of Gaussian (LoG):

![Laplacian of Gaussian (LoG)](https://github.com/jasonyeong/CariesDetection/blob/master/Blob%20Detection/README_img/Laplacian%20of%20Gaussian%20(LoG).jpg?raw=true "Laplacian of Gaussian (LoG)")

>This is the slowest but most accurate algorithm.
>It first applies a series of Gaussian filters at different scales, then performs a Laplacian operation on the filtered images. The local maximums in the combined image are the blobs to be detected.

### Preprocessing:

120 dental X-ray images are enhanced through rotation, scaling, and resizing, resulting in 11,114 images. These images are also preprocessed with gray image scaling and blurring.

### System flow diagram:

![System flow diagram](https://www.researchgate.net/publication/355065789/figure/fig1/AS:1076383186984960@1633641158198/System-flow-diagram_W640.jpg "System flow diagram")

>Feature Extraction:
>Noise Reduction(Gaussian filters)
>Geometric features enhanced(Hessian analysis)

