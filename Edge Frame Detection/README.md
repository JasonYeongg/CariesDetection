# Simple CNN

## Result on Caries

| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 12 | 501 | 3 | 13 | 48% | 80% | 60% | 96.98% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/result/result.jpg?raw=true "Result")

## Introduction of Each File

> process/urljson.py

Converts data from URLs into JSON files to avoid interruptions when generating training data.

> process/datahandle.py

Extracts masks of teeth, alveolar bone, and caries from JSON data, then crops training data based on these masks after augmentation.

> process/data_input.py

Preprocesses and normalizes training data when training with model/train.py (automatically called by train.py, no need for manual invocation).

> model/train.py

Works with network.py and process/data_input.py for training. Also handles oversampling and augmentation of different amounts of normal and caries data.

> predict/predict.py

Tests the folded models using the split testing data from model/train.py and applies Non-Maximum Suppression (NMS) to process the results.

## Introduction to the Edge Frame Detection

### Data Processing:

![1](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/1.jpg?raw=true "1")
![2](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/2.jpg?raw=true "2")
> To avoid noise and parts of the tooth root on both sides, edge reduction was applied to the teeth.

### Data Cropping:

![3](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/3.jpg?raw=true "3")
> The average width of each tooth was calculated to determine the appropriate bounding box size, and different training data was generated through scaling.

![4](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/4.jpg?raw=true "4")
> For caries data, the size of the caries within each bounding box and the proportion of the caries covered in the entire lesion vary, affecting training. Therefore, different sample weights were assigned to adjust the training effect.

| Caries Area in Box | Proportion of Caries in Lesion Area | Sample Weight | Label |
|:----------:|:----------:|:----------:|:----------:|
| >= 40% | >= 40% | 1.5 | Caries |
| >= 5% | >= 40% | 1 | Caries |
| >= 40% | >= 5% | 0.6 | Caries |
| >= 5% | >= 5% | 0.1 | Caries |
| >= 1% | >= 1% | 0.05 | Caries |
| <  1% | <  1% | 0.95 | Normal |

![5](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/5.jpg?raw=true "5")
> Since most caries occur on the sides of teeth, side training is enhanced by reducing the sample weight for top data, based on Hough Line Detection, by 0.5.

### Program Flow Diagram:

```mermaid
flowchart TD
subgraph datahandle.py
A([Original X-ray]) --> F[X-ray after noise masking]
B([Tissue Classification]) --> C[Tooth Mask]
B --> D[Caries Mask]
B --> E[Alveolar Bone Mask]
C --> G[Calculate average width of each tooth]
C -->|Mask areas outside the masks\nwith black to reduce noise interference| F
C -->|Use erosion and canny to\nextract edges for bounding data| H[Edge for Bounding Data]
G -->|Generate bounding boxes of different sizes\nfor bounding data| I[Bounding Data]
E -->|If most of the mask is detected,\ndiscard the bounding box| I
F --> J[Training Images]
H --> J
I -->|Calculate overlap between bounding boxes\nand set a limit| J
C -->|Find tangent lines close to horizontal\nusing HoughLines| K[Tooth Tangent Lines]
I --> L[Sample Weight]
K -->|Assign 0.5 Sample Weight to training data\ncontaining tooth tangent lines| L
D -->|Use IOU to calculate the size of caries\nbounded by the training data\nand assign different Sample Weights| L
end
subgraph data_input.py
J --> J1[Training Images]
J1 -->|Standardize image sizes and normalize,\ndiscard 30% of normal data randomly| M[Processed Training Images]
J1 -->|Assign labels based on contents of bounding box\nand apply to_categorical| N[Label]
M -->|ImageDataGenerator applied for\nrotation and flip augmentation| O[Augmented Images]
end
subgraph train.py
O --> O1[Augmented Images]
N --> N1[Label]
L --> L1[Sample Weight]
O1 --> P0[(Training Data)]
N1 --> P0
L1 --> P0
P0 --> P{Split Training Data by KFold}
P --> Q[/Oversampling: Randomly replicate minority class\nuntil balanced with majority class\]
Q --> Q1[Training Data]
Q --> Q2[Validation Data]
Q --> Q3[Testing Data]
end
subgraph network.py
Q1 --> R[(SimpleCNN)]
Q2 --> R[(SimpleCNN)]
R --> S([Model])
end
subgraph predict.py
S --> S1([Predict])
Q3 --> S1
S1 --> |Obtain results through max and argmax| SA[Prediction Results]
SA --> |Apply NMS to eliminate duplicates\nand low score results| SR[Caries Detection]
end
```
