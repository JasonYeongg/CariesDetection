# YoloV3
[Diagnosis of interproximal caries lesions with deep convolutional neural network in digital bitewing radiographs](https://www.researchgate.net/publication/352757579_Diagnosis_of_interproximal_caries_lesions_with_deep_convolutional_neural_network_in_digital_bitewing_radiographs "link")

## Result on Caries

|  | TP | FN | include | empty(FP) |  Sensitivity |   Precision |   F1 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ClassProbThreshold > 0.01 | 374 | 258 | 117 | 146 | 59.2% | 71.9% | 64.9% |
| ClassProbThreshold > 0.5 | 257 | 161 | 100 | 163 | 61.5% | 61.2% | 61.3% |
| IOU > 0.5 | 101 | 64 | 98 | 165 | 61.2% | 38% | 46.9% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/YoloV3/Result.jpg?raw=true "Result")


## Introduction of each file

> pross.py

將url轉換成json，再把caries和mask隨機做augmentation並存成jpg和txt後再隨機區分training和validation資料

> 2yolo.py

將資料轉換成可以提供Yolo訓練的模式

> detect.py

利用訓練好的model以及分割好的validation資料做測試

> Caries/caries.data

classes數量以及資料路徑

> Caries/classes.names

classes名稱

> Caries/darknet-yolov3.cfg

訓練相關參數

## Introduction to the YoloV3
[相關教程](https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/ "link")

## Introduction to the paper

### Dataset:	

* 800 (train)
* 200 (test & validation)
> (Did not use any image enhancement or pre-processing methods to improve the bitewing images)

### Augmentation:	

* Rotation, scaling, zooming, and cropping operations

### YoloV3:

![YoloV3 System](https://www.researchgate.net/publication/352757579/figure/fig4/AS:1116690653954049@1643251207096/Architecture-of-the-YOLO-based-CAA-system.png "YoloV3 System")

### Caries Detection Process:

![Evaluation approach](https://www.researchgate.net/publication/352757579/figure/fig3/AS:1038815154171914@1624684241943/Scheme-of-the-proposed-YOLO-based-CAA-system_W640.jpg "Evaluation approach")

![Evaluation approach](https://www.researchgate.net/publication/352757579/figure/fig5/AS:1038815170936832@1624684245242/Evaluation-approach-of-the-proposed-system_W640.jpg "Evaluation approach")

### Result:

![Result](https://www.researchgate.net/publication/352757579/figure/fig7/AS:1038815187697669@1624684249625/The-green-box-represents-the-ground-truth-while-the-orange-box-represents-the-detections_W640.jpg "Result")

> Green box: Ground truth; Orange box: Detections of proposed CNN model. 

> a, b, and c successful detections. 

> d, e, and f successful and false negative detections (only green boxes alone)
