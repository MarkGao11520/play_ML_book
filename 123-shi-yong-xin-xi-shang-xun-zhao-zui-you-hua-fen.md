# 12.3 使用信息熵寻找最优划分



```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:,2:]
y = iris.target
```


```python
# 引入决策树
from sklearn.tree import DecisionTreeClassifier

dt_cfl = DecisionTreeClassifier(max_depth=2, criterion="entropy")
dt_cfl.fit(X, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
def plot_decision_boundary(model,axis):
    """
    model：模型
    axis:坐标轴的范围；0123对应的就是x轴和y轴的范围
    """
    # 使用linspace将x轴，y轴划分成无数的小点
    x0,x1 = np.meshgrid(
        np.linspace(axis[0],axis[1],int((axis[1]-axis[0])*100)).reshape(-1,1),
        np.linspace(axis[2],axis[3],int((axis[3]-axis[2])*100)).reshape(-1,1)
    )
    X_new = np.c_[x0.ravel(),x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    plt.contourf(x0,x1,zz,linspace=5,cmap=custom_cmap)
```


```python
plot_decision_boundary(dt_cfl, axis=[0.5, 7.5, 0, 3])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x119db79b0>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-00d42888e6aa0c9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




### 模拟使用信息熵进行划分


```python
def spilt(X, y, d, value):
    index_a = (X[:,d] <= value)
    index_b = (X[:,d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]
```


```python
from collections import Counter
from math import log

def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p*log(p)
    return res

def try_spilt(X, y):
    best_entropy = float('inf')
    best_d, best_v = -1,-1
    for d in range(X.shape[1]):
        # 遍历每一个维度
        sorted_index = np.argsort(X[:,d])
        for i in range(1,len(X)):
            # 计算每两个值之间的信息熵，所以从1开始
            if X[sorted_index[i-1],d] != X[sorted_index[i],d]:
                # 将两个数据点的平均值作为切分点
                v = (X[sorted_index[i-1],d] + X[sorted_index[i],d]) / 2
                X_l, X_r, y_l, y_r = spilt(X, y, d, v)
                # 分别计算两部分的信息熵然后相加
                e = entropy(y_l) + entropy(y_r)
                if e<best_entropy:
                    best_entropy, best_d, best_v = e, d, v
    return best_entropy, best_d, best_v
```


```python
best_entropy, best_d, best_v = try_spilt(X, y)
print(best_entropy)
print(best_d)
print(best_v)
# 对比sklearn的划分结果，就是在横轴上（第0个维度），大约2.45的位置进行了划分
```

    0.6931471805599453
    0
    2.45



```python
X1_l, X1_r, y1_l, y1_r = spilt(X, y ,best_d, best_v)
```


```python
entropy(y1_l)
```




    0.0




```python
entropy(y1_r)
```




    0.6931471805599453




```python
best_entropy2, best_d2, best_v2 = try_spilt(x1_r, y1_r)
print(best_entropy2)
print(best_d2)
print(best_v2)
# 在第一个维度0.75的位置进行划分，划分的结果的信息熵为0.41左右
```

    0.4132278899361904
    1
    1.75



```python
X2_l, x2_r, y2_l, y2_r = spilt(X1_r,y1_r,best_d2,best_v2)
```


```python
entropy(y2_l)
```




    0.30849545083110386




```python
entropy(y2_r)
```




    0.10473243910508653


