# SOLOV2
Joey所分享的影像分割模型

## Result on Caries

| TP | FN | include | empty(FP) | Sensitivity | Precision | F1 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 318 | 559 | 443 | 0 | 36.3% | 100% | 53.22% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/SoloV2/caries/result.jpg?raw=true "Result")

## Introduction of each file

> caries

文件夾裡面包含了在SoloV2上效果最好的model以及訓練資料

> urljson.py

將url上的資料都轉成json文件，以避免生成訓練資料時中斷

> train.py

將資料隨機分類成training和validation後就開始訓練

> predict.py

用來在訓練完後做predict

[SoloV2的使用說明](https://docs.google.com/presentation/d/1ZeRDzgs-P2y4XMrfJyApH0MwY1YIS0gWhdT-39cS1dw/edit?usp=sharing "link")


