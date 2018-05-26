# 8.LASSO

### 使用|θ|代替θ<sup>2</sup>来标示θ的大小

![image.png](https://upload-images.jianshu.io/upload_images/7220971-97121b2768f4e717.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### Selection Operator -- 选择运算符
LASSO回归有一些选择的功能
![image.png](https://upload-images.jianshu.io/upload_images/7220971-451c364ada908520.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##1. 实际编程（准备代码参考上一节岭回归）

```python
from sklearn.linear_model import Lasso

def LassoRegression(degree,alpha):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scatter",StandardScaler()),
        ("lasso_reg",Lasso(alpha=alpha))
    ])
```


```python
# 这里穿的alpha起始值比岭回归的时候大了很多，是由于现在是绝对值
lasso1_reg = LassoRegression(degree=20,alpha=0.01)
lasso1_reg.fit(X_train,y_train)
lasso1_predict = lasso1_reg.predict(X_test)
mean_squared_error(lasso1_predict,y_test)
```




    1.1496080843259968




```python
plot_module(lasso1_reg)
```


![image.png](https://upload-images.jianshu.io/upload_images/7220971-4f6077259f57f014.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




```python
# 增大alpha继续试验
lasso2_reg = LassoRegression(degree=20,alpha=0.1)
lasso2_reg.fit(X_train,y_train)
lasso2_predict = lasso2_reg.predict(X_test)
mean_squared_error(lasso2_predict,y_test)
```




    1.1213911351818648




```python
# 非常接近一根直线
plot_module(lasso2_reg)
```

![image.png](https://upload-images.jianshu.io/upload_images/7220971-4a2b560f0b580785.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




```python
# 增大alpha继续试验
lasso3_reg = LassoRegression(degree=20,alpha=1)
lasso3_reg.fit(X_train,y_train)
lasso3_predict = lasso3_reg.predict(X_test)
mean_squared_error(lasso3_predict,y_test)
```




    1.8408939659515595




```python
# alpha=1的时候正则化已经过头了
plot_module(lasso3_reg)
```


![image.png](https://upload-images.jianshu.io/upload_images/7220971-f48fe5f89f284517.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




##2. 总结Ridge和Lasso
![image.png](https://upload-images.jianshu.io/upload_images/7220971-7e67b6c633e95f51.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
α=100的时候，使用Ridge的得到的模型曲线依旧是一根曲线，事实上，使用Ridge很难得到一根倾斜的直线，他一直是弯曲的形状

但是使用LASSO的时候，当α=0.1，虽然得到的依然是一根曲线，但是他显然比Radge的程度更低，更像一根直线

这是因为**LASSO趋向于使得一部分theta值为0（而不是很小的值），所以可以作为特征选择用**，LASSO的最后两个字母SO就是Selection Operator的首字母缩写
使用LASSO的过程如果某一项θ等于0了，就说明LASSO Regression认为这个θ对应的特征是没有用的，剩下的那些不等于0的θ就说明LASSO Regression认为对应的这些特征有用，所以他可以当做特征选择用

-------------------
> 当使用Ridge的时候，当α趋近与无穷大，那么使用梯度下降法的J(θ)的导数如下图，J(θ)向0趋近的过程中，每个θ都是有值的

![image.png](https://upload-images.jianshu.io/upload_images/7220971-2a3b9c8b64ab5e7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


> 但是LASSO不同,在LASSO的损失函数中，如果我们让α趋近于无穷，只看后面一部分的话，那么后面一部分的绝对值实际上是不可导的，我们可以使用一种sign函数刻画一下绝对值导数，如下图。那么这个时候，同样在J(θ)向0趋近的过程中，他会先走到θ等于0的y轴位置，然后再沿着y轴往下向零点的方向走

![image.png](https://upload-images.jianshu.io/upload_images/7220971-2d4e04e36068a5f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这也说明了Ridge为什么叫岭回归，因为他更像是翻山越岭一样，在梯度下降法中一点一点找坡度缓的方向前进。而LASSO的路径就比较规则，会在训练的过程中碰到一些轴使得某些θ为0。

所以从计算准确度上来说，我们应该更加倾向于Ridge，但是如果我们的维度比较多，样本非常大（比如多项式回归时degree=100）

