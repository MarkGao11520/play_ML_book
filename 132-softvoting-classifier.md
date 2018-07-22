# 13.2 Softvoting Classifier

更合理的投票，应该有权值

- Hard Voting
![image.png](https://upload-images.jianshu.io/upload_images/7220971-a97ab6204286540b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- Soft Voting
![image.png](https://upload-images.jianshu.io/upload_images/7220971-5376d76f90edf956.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
要求集合的每一个模型都能估算概率
    - 逻辑回归可以估算概率
    - KNN可以估算规律，结果占离他最近的k个点
    - 决策树，叶子结点中占比例最大的类别数据占整个叶子结点量的比值
    - SVM算法：probability: boolean, optional(default=False)-> True