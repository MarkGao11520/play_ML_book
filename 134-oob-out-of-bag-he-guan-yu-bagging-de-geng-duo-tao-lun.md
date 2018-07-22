# 13.4 oob (Out-of-Bag) 和关于Bagging的更多讨论

放回取样导致一部分样本很有可能没有取到

平均大约有37%的样本没有取到。这些样本就叫Out-of-Bag.

不使用测试数据集，而使用这部分没有取到的样本做测试/验证。

sikit-learn：oob\_score\_
```python
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500,
                                max_samples=100,
                                bootstrap=True,
                                oob_score=True)
bagging_clf.fit(X, y)
bagging_clf.oob_score_

0.92
```

Bagging的思路极易并行化处理

sikit-learn中对于并行处理的算法，可以传入n_jobs来调整使用几个核来处理

```python
```