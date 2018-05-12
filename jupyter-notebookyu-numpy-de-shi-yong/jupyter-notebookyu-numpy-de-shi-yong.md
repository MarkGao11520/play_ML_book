## 1.%run
- ### %run 执行python脚本，并将脚本中的函数加载
```
%run ./hello.py
```

- ### 可以直接使用import命令导入本机目录下的包
```
import mymodule.FirstML
```

## 2.%timeit
- ### %timeit 测试代码的性能
![2.1](https://upload-images.jianshu.io/upload_images/7220971-e93e6afb7085e16c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
测试结果表明，运行了一千次，取有价值的7次，平均每次耗时324+/-5.7 μs（有多少次循环是由Jupyter Notebook自动决定的）
- ### %%timeit 测试整个代码块
![2.2](https://upload-images.jianshu.io/upload_images/7220971-5e6b67773019858e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 3.%time
- ### 使用%time让测试只执行一次
![3.1](https://upload-images.jianshu.io/upload_images/7220971-453414dc9eb213dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
本次测试时间比上面的测试时间会多，是因为只测试了一次。可能不够准确

- ### 一个陷阱
使用%timeit 测试多次在每次测试的执行性能不一样的时候测试结果会不准确。
考虑用%timeit 测试一个排序算法，由于第一次执行完毕后数组已经排好序，那么在后面执行的时候，如果使用插入排序等算法就会导致后面999次的时间非常短，导致测试值不准确
![3.2](https://upload-images.jianshu.io/upload_images/7220971-a988f77dc4e3d164.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 4.其他
- 使用%lsmagic查看所有的魔法命令
![4.1](https://upload-images.jianshu.io/upload_images/7220971-495fb2a0de7e9e86.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 在使用方法后面加?查看文档
![4.1](https://upload-images.jianshu.io/upload_images/7220971-4fc01fa286717ca0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)