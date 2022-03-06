# Simple CNN

## Result on Caries

| TP | TN | FP | FN | Sensitivity | Precision | F1 | Accuracy |
|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| 12 | 501 | 3 | 13 | 48% | 80% | 60% | 96.98% |

![Result](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/result/result.jpg?raw=true "Result")

## Introduction of each file

> process/urljson.py

將url上的資料都轉成json文件，以避免生成訓練資料時中斷

> process/datahandle.py

通過json取得牙齒，齒槽骨和蛀牙的等mask取出，並依照這些mask在augmentation後crop出訓練用的資料

> process/data_input.py

用model/train.py訓練時將訓練資料做預處理以及歸一化等，再用做訓練(train.py會直接使用，不需要呼叫)

> model/train.py

配合network.py以及process/data_input.py直接訓練，並在這裡對不同數量的normal和caries資料做oversampling以及augmentation

> predict/predict.py

利用model/train.py分割好的testing資料，以及生成的folded model個別做測試，並以nms做處理

## Introduction to the Edge Frame Detection

### 資料處理:	

![1](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/1.jpg?raw=true "1")
![2](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/2.jpg?raw=true "2")
> 為了避免框到兩側的雜訊以及牙根的部分，對牙齒的邊緣做了刪減的處理

### 資料框選:	

![3](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/3.jpg?raw=true "3")
> 通過計算每顆牙齒的平均寬度，找出適合他的框的大小，再通過放大及縮小去產生不同的訓練資料

![4](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/4.jpg?raw=true "4")
> 針對Caries資料的部分，每一個框裡蛀牙所佔的大小，以及被框到的蛀牙佔一整塊蛀牙的比例都不一樣，會對訓練造成影響，所以在這裡我們需要給予不同的資料各自的Sample Weight來調整訓練效果 

| 框內蛀牙面積 | 蛀牙在蛀牙區域的面積 | Sample Weight | Label |
|:----------:|:----------:|:----------:|:----------:|
| >= 40% | >= 40% | 1.5 | Caries |
| >= 5% | >= 40% | 1 | Caries |
| >= 40% | >= 5% | 0.6 | Caries |
| >= 5% | >= 5% | 0.1 | Caries |
| >= 1% | >= 1% | 0.05 | Caries |
| <  1% | <  1% | 0.95 | Normal |

![5](https://github.com/jasonyeong/CariesDetection/blob/master/Edge%20Frame%20Detection/README_IMG/5.jpg?raw=true "5")
> 因為大部分Caries都是在牙齒兩側，為了加強兩側的訓練，我們在給予資料Sample Weight時按照Hough Line Detection得到的頂部資料乘上0.5以達到削弱的效果

### 程式流程圖:

```mermaid
flowchart TD
subgraph datahandle.py
A([X光片原圖]) --> F[X光片屏蔽雜訊後]
B([組織分類]) --> C[牙齒Mask]
B --> D[蛀牙Mask]
B --> E[齒槽骨Mask]
C --> G[計算每棵牙齒平均寬度]
C -->|將Mask以外的地方 \n覆蓋上黑色的Mask \n減少影響訓練的雜訊| F
C -->|利用erode和canny \n取得框選資料用的用的edge| H[資料框選線]
G -->|通過放大和縮小產生不同尺寸 \n用於框選資料的框| I[資料框]
E -->|偵測到大部分該Mask就取消該框| I
F --> J[訓練用圖片]
I -->|計算每個框之間的overlap\n並給予上限| J
H --> J
C -->|通過HoughLines尋找接近水平的切線| K[牙齒切線]
J --> L[SampleWeight]
K -->|包含牙齒切線的訓練資料\n給予0.5的SampleWeight| L
D -->|通過IOU計算訓練用資料框選到的\n蛀牙大小給予個別不同的SampleWeight| L
end
subgraph data_input.py
J --> J1[訓練用圖片]
J1 -->|统一图片尺寸並標準化\n每張資料,並隨即淘汰30%的\nnormal資料| M[處理後訓練圖片]
J1 -->|依照框到的內容決定\nlabel後再做\nto_categorical處理| N[label]
M -->|圖片經過ImageDataGenerator\n做rotation和flip| O[增強處理後的圖片]
end
subgraph train.py
O --> O1[增強處理後的圖片]
N --> N1[label]
L --> L1[SampleWeight]
O1 --> P{將訓練資料按KFold分配}
N1 --> P
L1 --> P
P --> Q[/Oversampling:通過不斷隨機複製\n數量少的一方直到與另一方一樣數量\]
Q --> Q1[TrainingData]
Q --> Q2[ValidationData]
Q --> Q3[TestingData]
end
subgraph network.py
Q1 --> R[(SimpleCNN)]
Q2 --> R[(SimpleCNN)]
R --> S([Model])
end
subgraph predict.py
S --> S1([Model])
S1 --> |通過max和armax取得model給出的結果| SA[Predict結果]
Q3 --> SA
SA --> |將predict的結果通過NMS做處理,\n淘汰重複且分數低的Result| SR[蛀牙辨識]
end
```
