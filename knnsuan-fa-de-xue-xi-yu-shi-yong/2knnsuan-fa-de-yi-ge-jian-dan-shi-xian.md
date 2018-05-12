```python
import numpy as np
import matplotlib.pyplot as plt
```

#### 原始集合


```
# 特征
raw_data_x= [[3.393533211,2.331273381],
             [2.110073483,1.781539638],
             [1.343808831,3.368360954],
             [3.582294042,4.679179110],
             [2.280362439,2.866990263],
             [7.423436942,4.696522875],
             [5.745051997,3.533989803],
             [9.172168622,2.511101045],
             [7.792783481,3.424088941],
             [7.939820817,0.791637231]
            ]
# 所述类别
raw_data_y = [0,0,0,0,0,1,1,1,1,1]
```

#### 训练集合


```python
X_train = np.array(raw_data_x)
y_train = np.array(raw_data_y)
# 要预测的点
x = np.array([8.093607318,3.365731514])
```

#### 绘制数据集及要预测的点


```python
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='g')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='r')
plt.scatter(x[0],x[1],color='b')
```




    <matplotlib.collections.PathCollection at 0x11addb908>




![1](https://upload-images.jianshu.io/upload_images/7220971-e83a2186a3fcfa39.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#### KNN 实现过程简单编码


```python
from math import sqrt
distances = []
for x_train in X_train:
    # 欧拉 
    # **2 求平方
    d = sqrt(np.sum((x_train - x)**2))
    distances.append(d)
distances
```




    [4.812566907609877,
     6.189696362066091,
     6.749798999160064,
     4.6986266144110695,
     5.83460014556857,
     1.4900114024329525,
     2.354574897431513,
     1.3761132675144652,
     0.3064319992975,
     2.5786840957478887]




```python
# 生成表达式
distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
distances
```




    [4.812566907609877,
     6.189696362066091,
     6.749798999160064,
     4.6986266144110695,
     5.83460014556857,
     1.4900114024329525,
     2.354574897431513,
     1.3761132675144652,
     0.3064319992975,
     2.5786840957478887]




```python
# 返回排序后的结果的索引,也就是距离测试点距离最近的点的排序坐标数组
nearset = np.argsort(distances)
```


```python
k = 6
```

### 投票


```python
# 求出距离测试点最近的6个点的类别
topK_y = [y_train[i] for i in nearset[:k]]
topK_y
```




    [1, 1, 1, 1, 1, 0]




```python
# collections的Counter方法可以求出一个数组的相同元素的个数，返回一个dict【key=元素名，value=元素个数】
from collections import Counter
Counter(topK_y)
```




    Counter({0: 1, 1: 5})




```python
# most_common方法求出最多的元素对应的那个键值对
votes = Counter(topK_y)
votes.most_common(1)
```




    [(1, 5)]




```python
 votes.most_common(1)[0][0]
```




    1




```python
predict_y = votes.most_common(1)[0][0]
predict_y
```




    1


----------
## KNN算法的封装
```
import numpy as np
from math import sqrt
from collections import Counter
class KNNClassifier:

    def __init__(self,k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回标示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "mush fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], \
            "the feature number of x must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        distances = [sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearset = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearset[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

```

## 再看机器学习
![1](https://upload-images.jianshu.io/upload_images/7220971-244643ec6ffb41eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

      可以说kNN是一个不需要训练过程的算法
      k近邻算法是非常特殊的，可以被认为是没有模型的算法
      为了和其他算法统一，可以认为训练数据集就是模型