# 12.2 信息熵

![image.png](https://upload-images.jianshu.io/upload_images/7220971-29e75c6772c29ad8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> 熵在信息论中代表随机变量中不确定度的度量。
    - 熵越大，数据的不确定性越高
    - 熵越小，数据的不确定性越低
    
![image.png](https://upload-images.jianshu.io/upload_images/7220971-01adc0662653bc84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/7220971-b23dfd78af5be264.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 二分类的信息熵函数


![image.png](https://upload-images.jianshu.io/upload_images/7220971-6c0473534603274d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def entropy(p):
    return -p * np.log(p) - (1-p) * np.log(1-p)
```


```python
x = np.linspace(0.01, 0.99, 200)

plt.scatter(x, entropy(x))
```




    <matplotlib.collections.PathCollection at 0x113b2e358>





![image.png](https://upload-images.jianshu.io/upload_images/7220971-d9d03e20f1a5c17e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


当x=0.5 的时候，信息熵达到了最大值（当两类样本各占一半的时候），也就是说这个时候的样本是最不稳定的

当系统每一个类别都是等概率的时候，其实是他最不确定的时候，此时他的信息熵是最高的。如果系统偏向于某一类，相当于有了约定向，信息熵逐渐降低，知道有一个类别占到了百分之百，此时信息熵达到最低值0


![image.png](https://upload-images.jianshu.io/upload_images/7220971-e07447acb3ce972b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

了解了信息熵的概念，解决上面的两个问题就好说了。在每一个结点上，都希望在某一个维度上基于某一个阈值进行划分，在划分以后要做的事情就是要让我们的数据划分成两部分之后，相应我们的系统整体的信息熵降低（也就是让我们的系统变的更加确定）

接下来的任务，就是找每个节点上游一个维度，在这个维度上有一个取值，根据这个取值进行划分，划分后是所有其他划分方式的信息熵中的最小值。我们就成当前的划分方式就是一个最好的划分。找到这个划分的方法就是对所有的可能性进行搜索。