![2](https://upload-images.jianshu.io/upload_images/7220971-a3e23b6467c6ec45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### train test split
![3](https://upload-images.jianshu.io/upload_images/7220971-8807e733e290a725.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 封装我们自己的 train test split


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
```


```python
# 加载鸢尾花数据集
iris = datasets.load_iris()
```


```python
X = iris.data
y = iris.target
```


```python
x.shape
```




    (150, 4)




```python
y.shape
```




    (150,)



### train_test_spilt


```python
# permutation(n) 给出从0到n-1的一个随机排列
shuffle_indexes = np.random.permutation(len(X))
```


```python
shuffle_indexes
```




    array([139,  40,  63, 138,  88, 123, 101, 122,  89,   0, 132, 108, 120,
           111, 140,  30,  47,   6, 128,  46,  49, 105,   3,  53,  85,   9,
           147,  95, 116,  75,  20, 134,  34,  42, 144,   7,  10,  73,  90,
            72, 141,  99,  57,  93,  74, 103,  39, 106,  86,  35,  15,  96,
            78, 129,  19,  51, 117,  62, 113,  77, 100, 118,  83,  18,  70,
            94,  26,  25,  12,  50,  28, 133, 145,  43,  33, 109,  44, 114,
            92, 112,  82, 119, 115,  69,  27,  80,  41,  38,  98,  97,  61,
            16,  56,  11,  64, 135,   1, 126, 137,  45,  32,  60, 124,  71,
            58,  52,  84,  21,  81,  13, 142, 127,  55,  79,  14,  68, 146,
            48,  23,  76,  17,   8, 136, 110,  87,   2, 143, 104,  24,  37,
           107,  31,   4, 131,  66, 121, 149, 102,   5,  65,  54, 148,  59,
           125,  29,  67,  36,  91, 130,  22])




```python
# 测试数据集的比例
test_ratio = 0.2
# 获取测试数据集
tets_size = int(len(X) * test_ratio)
tets_size
```




    30




```python
test_indexes = shuffle_indexes[:tets_size]
train_indexes = shuffle_indexes[tets_size:]
```


```python
X_train = X[train_indexes]
y_train = y[train_indexes]

X_test = X[test_indexes]
y_test = y[test_indexes]
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (113, 4)
    (113,)
    (37, 4)
    (37,)


### 使用我们自己封装的测试分割函数分割训练集

```
import numpy as np


def train_test_split(X, y, test_radio=0.2, seed=None):
    """将数据X和y按照test_radio分割成X_train,y_train,X_test,y_test"""
    assert X.shape[0] == y.shape[0],\
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_radio <= 1.0, \
        "test_radio must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X)*test_radio)

    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test

```


```python
import machine_learning
from machine_learning.module_selection import train_test_split
```


```python
X_train,y_train,X_test,y_test = train_test_split(X,y,test_radio=0.25)
```

### 测试我们的KNN算法


```python
from machine_learning.KNN import KNNClassifier
```


```python
my_knn_clf = KNNClassifier(k=6)
```


```python
my_knn_clf.fit(X_train,y_train)
```




    <machine_learning.KNN.KNNClassifier at 0x1a102a3a58>




```python
# 预测结果
y_predict = my_knn_clf.predict(X_test)
```


```python
y_predict
```




    array([2, 2, 2, 1, 0, 0, 2, 2, 2, 1, 1, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2,
           0, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 2, 2])




```python
y_test
```




    array([2, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2,
           0, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 1, 2])




```python
# 求出准确率
sum(y_predict==y_test)/len(y_test)
```




    0.9473684210526315




```python
from sklearn.model_selection import train_test_split
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y)
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (112, 4)
    (112,)
    (38, 4)
    (38,)



```python
from sklearn.neighbors import KNeighborsClassifier
sklearn_knn_clf = KNeighborsClassifier(n_neighbors=6)
```


```python
sklearn_knn_clf.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=6, p=2,
               weights='uniform')




```python
y_predict = sklearn_knn_clf.predict(X_test)
```


```python
y_predict
```




    array([2, 2, 2, 1, 0, 0, 2, 2, 2, 1, 1, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2,
           0, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 2, 2])




```python
y_test
```




    array([2, 2, 2, 1, 0, 0, 2, 2, 2, 2, 1, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2,
           0, 2, 0, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 1, 2])




```python
sum(y_predict==y_test)/len(y_test)
```




    0.9473684210526315

