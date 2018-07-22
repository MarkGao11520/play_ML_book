# 13.1 什么是集成学习

![image.png](https://upload-images.jianshu.io/upload_images/7220971-8459558946e66918.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn import datasets

X, y = datasets.make_moons(n_samples=500, noise=0.3, random_state=42)
plt.scatter(X[y==0, 0],X[y==0, 1])
plt.scatter(X[y==1, 0],X[y==1, 1])
```




    <matplotlib.collections.PathCollection at 0x113f8ef98>




![image.png](https://upload-images.jianshu.io/upload_images/7220971-8fa61bbda96652de.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
```


```python
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()
log_clf.fit(X_train, y_train)
log_clf.score(X_test, y_test)
```




    0.824




```python
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.score(X_test, y_test)
```




    0.88




```python
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)
```




    0.84




```python
# 手动完成集成学习的过程
y_predict1 = log_clf.predict(X_test)
y_predict2 = svm_clf.predict(X_test)
y_predict3 = dt_clf.predict(X_test)

# 只有至少两个算法预测结果为1，三个相加才会大于2
y_predict = np.array((y_predict1+y_predict2+y_predict3) >= 2, dtype='int')
```


```python
y_predict[:10]
```




    array([1, 1, 0, 0, 0, 1, 0, 1, 0, 1])




```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)
```




    0.896



### 使用Voting Classifier


```python
from sklearn.ensemble import VotingClassifier

# hard 少数服从多数
voting_clf = VotingClassifier(estimators=[
    ("log_clf", LogisticRegression()),
    ("svm_clf", SVC()),
    ("dt_clf", DecisionTreeClassifier())
], voting="hard")

voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
```

    /anaconda3/lib/python3.6/site-packages/sklearn/preprocessing/label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:





    0.88


