# YoloV3
[Diagnosis of interproximal caries lesions with deep convolutional neural network in digital bitewing radiographs](https://www.researchgate.net/publication/352757579_Diagnosis_of_interproximal_caries_lesions_with_deep_convolutional_neural_network_in_digital_bitewing_radiographs "link")

## Result on Caries

|  | TP | FN | Include | Empty (FP) | Sensitivity | Precision | F1 |
|:-----------------------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ClassProbThreshold > 0.01      | 374 | 258 | 117 | 146 | 59.2% | 71.9% | 64.9% |
| ClassProbThreshold > 0.5       | 257 | 161 | 100 | 163 | 61.5% | 61.2% | 61.3% |
| IOU > 0.5                      | 101 | 64  | 98  | 165 | 61.2% | 38%   | 46.9% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/YoloV3/Result.jpg?raw=true "Result")

## Introduction of Each File

> pross.py

Converts URLs into JSON, randomly augments caries and mask data, saves them as JPG and TXT files, then randomly splits the data into training and validation sets.

> 2yolo.py

Converts the data into a format compatible with Yolo for training.

> detect.py

Tests using the trained model and the split validation data.

> Caries/caries.data

Defines the number of classes and the data paths.

> Caries/classes.names

Lists the class names.

> Caries/darknet-yolov3.cfg

Contains the training parameters.

## Introduction to YoloV3
[Related Tutorial](https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/ "link")

## Introduction to the Paper

### Dataset:	

* 800 (train)
* 200 (test & validation)
> (No image enhancement or pre-processing was applied to improve the bitewing images)

### Augmentation:	

* Rotation, scaling, zooming, and cropping operations.

### YoloV3:

![YoloV3 System](https://www.researchgate.net/publication/352757579/figure/fig4/AS:1116690653954049@1643251207096/Architecture-of-the-YOLO-based-CAA-system.png "YoloV3 System")

### Caries Detection Process:

![Evaluation approach](https://www.researchgate.net/publication/352757579/figure/fig3/AS:1038815154171914@1624684241943/Scheme-of-the-proposed-YOLO-based-CAA-system_W640.jpg "Evaluation approach")

![Evaluation approach](https://www.researchgate.net/publication/352757579/figure/fig5/AS:1038815170936832@1624684245242/Evaluation-approach-of-the-proposed-system_W640.jpg "Evaluation approach")

### Result:

![Result](https://www.researchgate.net/publication/352757579/figure/fig7/AS:1038815187697669@1624684249625/The-green-box-represents-the-ground-truth-while-the-orange-box-represents-the-detections_W640.jpg "Result")

> Green box: Ground truth; Orange box: Detections of the proposed CNN model.

> a, b, and c: successful detections.

> d, e, and f: successful and false negative detections (only green boxes alone).
