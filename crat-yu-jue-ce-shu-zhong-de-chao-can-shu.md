# 12.5 CRAT 与决策树中的超参数

- CRAT： Classification And Regression Tree 
- 根据某一个维度d和某一个阈值v进行二分
- scikit-learn的决策实现：CRAT
- ID3，C4.5，C5.0等是使用其他的方式实现决策树

![image.png](https://upload-images.jianshu.io/upload_images/7220971-723e56461385f37b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

X, y = datasets.make_moons(noise=0.25, random_state=666)
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```




    <matplotlib.collections.PathCollection at 0x10bd5ee48>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-ea97b03aee5d98bc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
from sklearn.tree import DecisionTreeClassifier

# 默认使用基尼系数划分数据
# 不传max_depth会一直划分直到基尼系数为0为止
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
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
# 决策边界非常不规则，产生了过拟合
plot_decision_boundary(dt_clf, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x112f5e048>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-3559f9548166588e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
dt_clf2 = DecisionTreeClassifier(max_depth=2)
dt_clf2.fit(X, y)

# 很显然，现在这个样子相比上面的形状不在有过拟合，有了非常清晰的边界（不会针对某几个特别的样本点进行特殊的变化）
plot_decision_boundary(dt_clf2, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x1133499e8>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-af12930be028e365.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这种情况下，很有可能存在欠拟合，所以我们要对这些参数进行比较精细的调整，让他既不过拟合也不欠拟合


```python
# 对于一个结点来说，至少要有多少个样本数据，我们才对他继续进行拆分下去
dt_clf3 = DecisionTreeClassifier(min_samples_split=10)
dt_clf3.fit(X, y)

plot_decision_boundary(dt_clf3, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x113462f28>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-66c7fff3536d95e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
# 对于叶子节点来说，至少要有几个样本
dt_clf4 = DecisionTreeClassifier(min_samples_leaf=6)
dt_clf4.fit(X, y)

plot_decision_boundary(dt_clf4, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x1134babe0>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-6d46371587ba0356.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
# 决策树最多的叶子节点
dt_clf5 = DecisionTreeClassifier(max_leaf_nodes=4)
dt_clf5.fit(X, y)

plot_decision_boundary(dt_clf5, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
```

    /anaconda3/lib/python3.6/site-packages/matplotlib/contour.py:967: UserWarning: The following kwargs were not used by contour: 'linspace'
      s)





    <matplotlib.collections.PathCollection at 0x1136944e0>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-4d923aa0dbc9fdfb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![image.png](https://upload-images.jianshu.io/upload_images/7220971-b8775d89a3fed3f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

