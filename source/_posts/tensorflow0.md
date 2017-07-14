---
title: tensorflow学习：神经网络基本概念形象理解
tag: tensorflow学习
---

### softmax
softmax用做输出层，计算分类概率，见下图
<!-- more -->
就是把一堆实数的值映射到0-1区间，并且使他们的和为1。一般用来估计posterior probability，在多分类任务中有用到
为什么要取指数？第一个原因是要模拟max的行为，所以要让大的更大。第二个原因是需要一个可导的函数。

### sigmoid
模拟阶跃函数，首先映射到0不是0就是1，但是往往取中值0.5

sigmoid函数只能分两类，而softmax能分多类，softmax是sigmoid的扩展。

### 全连接层
参考了[知乎](https://www.zhihu.com/question/41037974)上的所有回答
卷积取的是局部特征，全连接就是把以前的局部特征重新通过权值矩阵组装成完整的图。因为用到了所有的局部特征，所以叫全连接。
fc层之前的conv层都只能concentrate on local region，也就是说如果你需要的是整张图片的high-level representation，加fc层就没错。但是如果你的目标是pixel-wise prediction，比如segmentation，那就大胆地去掉fc层换全*卷积*吧！具体可看fully convolutional network这篇paper～

还不是太理解，有待考证~
