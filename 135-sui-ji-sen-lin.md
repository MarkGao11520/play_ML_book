# 13.5 随机森林

Bagging

Base Estimator: Decision Tree

决策树在节点划分上，在随机的特征子集上寻找最优划分特征


```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, 
                                random_state=666, 
                                oob_score=True, 
                                n_jobs=-1)
rf_clf.fit(X, y)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=True, random_state=666, verbose=0, warm_start=False)




```python
rf_clf.oob_score_
```




    0.896




```python
# 随机森林包含决策树的参数
rf_clf2 = RandomForestClassifier(n_estimators=500, 
                                random_state=666, 
                                max_leaf_nodes=10,
                                oob_score=True, 
                                n_jobs=-1)
rf_clf2.fit(X, y)
rf_clf2.oob_score_
```




    0.912
    
### Extra-Trees 

决策树在节点划分上，使用随机的特征和随机的阈值(理论基础：根据BAGGING CLASSFIFER 的原理，只要大多数的决策树的决策能力比扔硬币的能力好一点就够了)
提供额外的随机性，抑制过拟合，但增大了bias
更快的训练速度

```
from sklearn.ensemble import ExtraTreesClassifier

et_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=666)
et_clf.fit(X, y)

ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='gini',
           max_depth=None, max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,
           oob_score=True, random_state=666, verbose=0, warm_start=False)
           
et_clf.oob_score_
0.892
```

