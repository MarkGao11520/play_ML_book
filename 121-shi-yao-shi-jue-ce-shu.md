# 12.1 什么是决策树

![image.png](https://upload-images.jianshu.io/upload_images/7220971-a9074f86b5cad8d9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



这样的一个招聘过程形成了一个树结构，这颗树的所有叶子结点其实就是我们最终的决策。也相对于是对与输入（录用者信息）的分类（录用/考察）。这样的一个过程，就是决策树。

对于这样的决策树来说，他具有数据结构利用树的所有性质。包括根结点，叶子结点，深度。在这里我们称决策树的深度为3



### 使用sklearn中的决策树直观的感受


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:,2:]
y = iris.target

plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])
```




    <matplotlib.collections.PathCollection at 0x10bae8630>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-01e9cb8fef97f702.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



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





    <matplotlib.collections.PathCollection at 0x10bf84358>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-00d42888e6aa0c9a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![image.png](https://upload-images.jianshu.io/upload_images/7220971-d98c57e6b4948618.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这就是决策树在面对属性是数值特征的时候是怎么处理的。
在每一个结点上，他首先选择一个维度以及这个维度上的一个阈值，分成两支，循环往复，来进行分类

> ### 总结
- 决策树是一个非参数学习的算法
- 决策树可以解决分类问题
- 天然的可以解决多分类问题
- 可以解决回归问题（将最终预测数据点落在叶子节点所有数据点的平均值作为预测值）
- 非常好的可解释性

> ### 构建决策树的问题 
- 每个结点在哪个维度做划分
- 某个维度在哪个值上做划分


