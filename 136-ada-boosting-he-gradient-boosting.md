# 13.6 Ada Boosting 和 Gradient Boosting

### Ada Boosting
集成多个模型
每个模型都在尝试增强(Boosting)整体的效果

![image.png](https://upload-images.jianshu.io/upload_images/7220971-2461acfc968b4b29.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

使用每次增强训练出来的子模型进行投票得出最终的结果


```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=500, random_state=666)
ada_clf.fit(X_train, y_train)


AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1.0, n_estimators=500, random_state=666)

ada_clf.score(X_test, y_test)
0.864
```

### Gradient Boosting
训练一个模型m1,产生错误e1
針対e1訓繚第二个模型m2,产生錯俣e2
针对e2训练第三个模型m3,产生错误e3..
最终预测结果是:m1 + m2 + m3 +