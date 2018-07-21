# 12.6 决策树解决回归问题

在决策树建立之后，每个叶子结点都包含了一些数据。
- 如果我们的数据输出的是类别的话，我们就让这些叶子结点包含的数据进行投票，票数最多的即为我们输出的分类
- 如果我们的数据输出的是数据的话，那就是回归问题所解决的问题，一个预测的数据来临之后，通过决策树到达了某一个叶子节点，我们就可以将这些叶子节点包含数据的平均值作为我们的预测值

### scikit-learn封装的决策树解决回归问题



```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 666)
```

### Decision Tree Regression


```python
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=None, splitter='best')




```python
dt_reg.score(X_test, y_test)
```




    0.5782292347434448




```python
# 在训练数据集上预测的准确率是百分之百的---过拟合
dt_reg.score(X_train, y_train)
```




    1.0


