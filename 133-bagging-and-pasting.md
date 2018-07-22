# 13.4 Bagging and Pasting

虽然有很多机器学习方法
但是从投票的角度看，还是不够多
创建更多的子模型！集成更多的子模型的意见
子模型之间不能一直！模型之间要有差异

如何创建差异性？
只看样本的一部分
例如: -共有500个样本数据;每个子模型只看100个样本数据，每个子模型不需要太高的准确率

如果每个子模型只有51%的准确率
![image.png](https://upload-images.jianshu.io/upload_images/7220971-56fe19eac1cdaeec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果每个子模型有60%的准确率
![image.png](https://upload-images.jianshu.io/upload_images/7220971-740ae9abeb36b983.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


取样：放回取样（Bagging），不放回取样（Pasting）
Bagging更常用

