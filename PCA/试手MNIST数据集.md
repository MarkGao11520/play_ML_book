# 7.试手MNIST数据集


## 1.加载MNIST数据集


```python
import numpy as np
from sklearn.datasets import fetch_mldata
```


```python
mnist = fetch_mldata("MNIST original")
```


```python
X,y = mnist.data,mnist.target
```


```python
X_train = np.array(X[:60000],dtype=float)
y_train = np.array(y[:60000],dtype=float)
X_test = np.array(X[60000:],dtype=float)
y_test = np.array(y[60000:],dtype=float)
```

## 2.使用KNN
> sklearn 封装的KNeighborsClassifier，在fit过程中如果数据集较大，会以树结构的过程进行存储，以加快knn的预测过程，但是会导致fit过程变慢
没有进行数据归一化，是因为这里的每个维度都标示的是每个像素点的亮度，他们的尺度是相同的，这个时候比较两个样本之间的距离是有意义的


```python
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
%time knn_clf.fit(X_train,y_train)
```

    CPU times: user 31.3 s, sys: 209 ms, total: 31.5 s
    Wall time: 31.7 s





    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')




```python
%time knn_clf.score(X_test,y_test)
```

    CPU times: user 10min 41s, sys: 2.55 s, total: 10min 44s
    Wall time: 10min 47s





    0.9688



## 3.PCA进行降维


```python
from sklearn.decomposition import PCA
# 使用可以解释百分之90原数据集的主成分
pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
# 从784维降到了87维，只用87维就可以解释百分之90的原数据集
X_train.shape
# (60000, 784)
X_train_reduction
```




    (60000, 784)




```python
X_test_reduction = pca.transform(X_test)
knn_clf = KNeighborsClassifier()
%time knn_clf.fit(X_train_reduction,y_train)
```

    CPU times: user 352 ms, sys: 2.54 ms, total: 355 ms
    Wall time: 356 ms





    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')



> 使用PCA进行降维后的数据集进行训练，不光时间变短了，准确度也变高了
这是因为PCA的过程中，不仅仅是进行了降维，还在降维的过程中将数据包含的噪音给消除了
这使得我们可以更加好的，更加准确的拿到我们数据集对应的特征，从而使得准确率大大提高


```python
%time knn_clf.score(X_test_reduction,y_test)
```

    CPU times: user 1min 2s, sys: 197 ms, total: 1min 2s
    Wall time: 1min 2s





    0.9728


