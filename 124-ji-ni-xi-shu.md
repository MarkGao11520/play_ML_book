# 12.4 基尼系数

基尼系数的意义和信息熵相同

![image.png](https://upload-images.jianshu.io/upload_images/7220971-b1a7ae2a8551cdee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


二分类基尼系数函数

![image.png](https://upload-images.jianshu.io/upload_images/7220971-fd6d905bc5450ccd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


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

dt_cfl = DecisionTreeClassifier(max_depth=2, criterion="gini")
dt_cfl.fit(X, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
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





    <matplotlib.collections.PathCollection at 0x1a11416c88>





![image.png](https://upload-images.jianshu.io/upload_images/7220971-00d42888e6aa0c9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 模拟使用基尼系数进行划分

只需要把信息熵的函数改成基尼系数


```python
def spilt(X, y, d, value):
    index_a = (X[:,d] <= value)
    index_b = (X[:,d] > value)
    return X[index_a], X[index_b], y[index_a], y[index_b]
```


```python
from collections import Counter
from math import log

def gini(y):
    counter = Counter(y)
    res = 1.0
    for num in counter.values():
        p = num / len(y)
        res -= p**2
    return res

def try_spilt(X, y):
    best_g = float('inf')
    best_d, best_v = -1,-1
    for d in range(X.shape[1]):
        # 遍历每一个维度
        sorted_index = np.argsort(X[:,d]) #在每一个特征维度上排序
        for i in range(1,len(X)):
            # 计算每两个值之间的信息熵，所以从1开始
            if X[sorted_index[i-1],d] != X[sorted_index[i],d]:
                # 将两个数据点的平均值作为切分点
                v = (X[sorted_index[i-1],d] + X[sorted_index[i],d]) / 2
                X_l, X_r, y_l, y_r = spilt(X, y, d, v)
                # 分别计算两部分的基尼系数然后相加
                e = gini(y_l) + gini(y_r)
                if e<best_g:
                    best_g, best_d, best_v = e, d, v
    return best_g, best_d, best_v
```


```python
best_g, best_d, best_v = try_spilt(X, y)
print(best_g)
print(best_d)
print(best_v)
```

    0.5
    0
    2.45



```python
X1_l, X1_r, y1_l, y1_r = spilt(X, y ,best_d, best_v)
```


```python
gini(y1_l)
```




    0.0




```python
gini(y1_r)
```




    0.5




```python
best_g2, best_d2, best_v2 = try_spilt(X1_r, y1_r)
print(best_g2)
print(best_d2)
print(best_v2)
```

    0.2105714900645938
    1
    1.75



```python
X2_l, x2_r, y2_l, y2_r = spilt(X1_r,y1_r,best_d2,best_v2)
```


```python
gini(y2_l)
```




    0.1680384087791495




```python
gini(y2_r)
```




    0.04253308128544431


>## 总结
- 熵信息的计算比基尼系数稍慢
- scikit-learn中默认使用基尼系数
- 大多数时候二者没有特别的效果优劣
