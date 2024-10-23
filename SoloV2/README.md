# SOLOV2

## Result on Caries

| TP | FN | Include | Empty (FP) | Sensitivity | Precision | F1 |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 318 | 559 | 443 | 0 | 36.3% | 100% | 53.22% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/SoloV2/caries/result.jpg?raw=true "Result")

## Introduction of Each File

> caries

This folder contains the best-performing model on SoloV2 along with the training data.

> urljson.py

Converts data from URLs into JSON files to avoid interruptions when generating training data.

> train.py

Randomly splits the data into training and validation sets, then begins training.

> predict.py

Used to make predictions after training is completed.
