# Python3入门机器学习-笔记整理


### 本笔记记录笔者观看[慕课网入门机器学习](https://coding.imooc.com/class/169.html)课程的笔记过程

### 官方Github代码：https://github.com/liuyubobobo/Play-with-Machine-Learning-Algorithms

主要以sikit-learn和numpy为技术栈，学习了机器学习入门的基本算法，并自己实现了部分sikit-learn中提供的算法


- #### 以下列出本笔记（课程）学习使用到的sikit-learn算法

- #### datasets可以用来加载真实数据进行模型训练的测试



```
import sklearn.datasets
datasets.load_iris() # 用于加载鸢尾花数据集
datasets.load_digits() # 用于加载手写识别的数据集
datasets.load_boston() #  用于加载波士顿房价的数据集

# fetch_mldata用于加载MNIST数据集
from sklearn.datasets import fetch_mldata

# fetch_lfw_people用于加载人脸数据集
from sklearn.datasets import fetch_lfw_people
```


- #### model_selection模块提供了模型选择的相关操作

```
# train_test_split用于分割测试数据集和训练数据集
from sklearn.model_selection import train_test_split

# GridSearchCV用于进行参数搜索，寻找合适的超参数
from sklearn.model_selection import GridSearchCV 
```

- #### metrics模块提供了数据之间的度量相关运算

```
# MSE的实现
from sklearn.metrics import mean_squared_error
# MAE的实现
from sklearn.metrics import mean_absolute_error
# r2_score的实现
from sklearn.metrics import r2_score
# scikit-learn中的混淆矩阵，精准率和召回率,f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
# scikit-learn中的Precision-Recall 曲线
from sklearn.metrics import precision_recall_curve
# scikit-learn中的ROC曲线
from sklearn.metrics import roc_curve,roc_auc_score
```

- #### multiclass模块提供了多分类问题的相关实现
    
```
# OneVsRestClassifier是OvR的实现
from sklearn.multiclass import OneVsRestClassifier

# OneVsOneClassifier是OvO的实现
from sklearn.multiclass import OneVsOneClassifier
```



- #### preprocessing模块提供了数据预处理的相关操作

```
# PolynomialFeatures进行多项式曾维处理，使用线性回归的方法解决非线性问题
from sklearn.preprocessing import PolynomialFeatures

# StandardScaler提供数据归一化运算
from sklearn.preprocessing import StandardScaler
```

- #### neighbors模块提供了近邻相关的算法实现

```
# KNeighborsClassifier是KNN算法解决分类问题的实现
from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier是KNN算法解决回归问题的实现
from sklearn.neighbors import KNeighborsRegressor
```


- #### linear_model提供了线性模型相关算法的实现

```
# LinearRegression是线性回归算法的实现
from sklearn.linear_model import LinearRegression

# SGDRegressor是梯度下降法相关的实现
from sklearn.linear_model import SGDRegressor

# Ridge是岭回归的实现
from sklearn.liner_model import Ridge

# LogisticRegression是逻辑回归的实现，默认使用了l2正则化
from sklearn.linear_model import LogisticRegression

```

- #### decomposition提供了降维相关算法的实现

```
# PCA给出了主成分分析法的相关实现
from sklearn.decomposition import PCA

```

- #### svm提供了支持向量机相关算法的实现

```
# SVC--Support Vector Classifier 使用支持向量机思想解决分类问题
from sklearn.svm import LinearSVC

# 使用核函数的支撑向量机
from sklearn.svm import SVC

# SCR--Support Vector Regression 使用支持向量机思想解决回归问题
from sklearn.svm import LinearSVR 

# 使用核函数的解决回归问题的支撑向量机
from sklearn.svm import SVR
```

