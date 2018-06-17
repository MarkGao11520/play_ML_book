# 9.4 F1 Score

## 1.F1 Score的含义
精准率和召回率是两个指标，有的时候精准率高一些，有的时候召回率高一些，在我们使用的时候，我们应该怎么解读这个精准率和召回率呢？
这个问题的答案，和机器学习大多数的取舍是一样的，应该视情况而定。

- ### 有的时候我们注重精准率，如股票预测。
> 我们希望这个比例越高越好。如果我们预测股票升了，我们就要购买这个股票，如果我们犯了FP的错误（实际上股票将下来了，而我们预测升上来了），那么我们就就亏钱了。 但是对于这个应用来说，很有可能我们对召回率不是特别关注。可能有很多上升周期，但是我们落掉了一些上升的周期（本来为1，我们错误的判断为0），这对我们没有太多的损失，因为我们漏掉了他，也不会投钱进去。

- ### 有的时候我们注重召回率，如病人诊断。
> 召回率低意味着:本来一个病人得病了，但是我们没有把他预测出来，这就意味着这个病人的病情会继续恶化下去。所以召回率更加重要，我们希望把所有有病的患者都预测出来。但是精准率却不是特别重要，因为本来一个人没病，我们预测他有病，这时候让他去做进一步的检查，进行确诊就好了。我们犯了FP的错误，只是让他多做了一次检查而已。这个时候召回率比精准率重要。

- ### 有的时候我们希望同时关注精准率和召回率，这个时候我们可以使用F1 Score，让我们兼顾精准率和召回率

![image.png](https://upload-images.jianshu.io/upload_images/7220971-f0d0f37ba134412c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

调和平均值的特点：如果一个值特别高，一个值特别低，那么我们得到的F1 Score 也将特别的低，只有二者都非常高，我们得到的值才会特别高

![image.png](https://upload-images.jianshu.io/upload_images/7220971-41b10877c1de8d8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 2.F1 Score 的实现

```python
import numpy as np
```


```python
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)
```


```python
precision = 0.5
recall = 0.5
f1_score(precision, recall)
# 当二者相同，得到的就是这个相同的值
```




    0.5




```python
precision = 0.1
recall = 0.9
f1_score(precision, recall)
# 有一个非常小，整体就非常小
```




    0.18000000000000002




```python
precision = 0.9
recall = 0
f1_score(precision, recall)
```




    0.0




```python
precision = 0.9
recall = 0.95
f1_score(precision, recall)
# 只有都非常大，结果才会打
```




    0.9243243243243242




```python
from sklearn import datasets
digits = datasets.load_digits()

X = digits.data
y = digits.target.copy()

# 手动将手写数据集变成及其偏斜的数据。不是9的y=0，是9的y=1
y[digits.target==9] = 1
y[digits.target!=9] = 0

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=666)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.score(X_test,y_test)

y_log_predict = log_reg.predict(X_test)
```


```python
from sklearn.metrics import f1_score
f1_score(y_test, y_log_predict)
```




    0.8674698795180723


0.8674698795180723 显然没有精准率和召回率高，这是因为首先我们的数据是有偏的，精准率和召回率都比准确率要低一些，在这里精准率和召回率能够更好的反应我们的结果。其次使用逻辑回归进行预测，明显召回率比较低，所以f1_score 被召回率拉低了，这个时候对于这个有偏的数据来说，我们更倾向认为f1_score 能更好的反应算法的水平