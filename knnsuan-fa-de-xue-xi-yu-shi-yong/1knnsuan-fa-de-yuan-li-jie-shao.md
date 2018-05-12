
- ###  优点
![优点](https://upload-images.jianshu.io/upload_images/7220971-ff3dbd9803ec8ebf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- ### 缺点
![缺点1](https://upload-images.jianshu.io/upload_images/7220971-bcfcb1ee918330cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![缺点2，3](https://upload-images.jianshu.io/upload_images/7220971-1544dda420356d81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![缺点4 维数灾难](https://upload-images.jianshu.io/upload_images/7220971-8665e9b426057666.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



- ### 原理案例介绍
![原理案例介绍](https://upload-images.jianshu.io/upload_images/7220971-5fefef4f338704a7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

假设现在设计一个程序判断一个新的肿瘤病人是良性肿瘤还是恶性肿瘤

先基于原有的肿瘤病人的发现时间和肿瘤大小（特征）对应的良性/恶性（值）建立了一张散点图，横坐标是肿瘤大小，纵坐标是发现时间，红色代表良性，蓝色代表恶性，现在要预测的病人的颜色为绿色

1. 首先需要取一个k值（这个k值的取法后面会介绍），然后找到距离要预测的病人的点（绿点）距离最近的k个点
2. 然后用第一步中取到的三个点进行投票，比如本例中投票结果就是```蓝：红 = 3：0``` ,3>0,所以判断这个新病人幻的事恶性肿瘤

- ### 本质
如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
