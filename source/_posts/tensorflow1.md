---
title: tensorflow学习： mnist 之softmax实现和CNN实现（1）
tag: tensorflow学习
---
### 一、softmax 
#### **基本概念**
#### **占位符**
一般用来做输入输出
<!-- more -->
#### **变量**
一个变量代表着TensorFlow计算图中的一个值
```
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

#### **类别预测与损失函数**
定义输出分类
```
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
指定损失函数
#### **训练模型**
1.设置迭代方法，步长
TensorFlow有大量内置的优化算法 这个例子中，我们用最速下降法让交叉熵下降，步长为0.01.

```
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

2.开始迭代
返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行train_step来完成。

```
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

### 评估模型
#### **计算误差**

```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
####  **构建Softmax 回归模型**
### 二、CNN
####  **卷积/池化的代码解析**
卷积，输入图片和卷积核

```
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') //x是图片，W是卷积核
```

Pooling是取矩阵中的最大值

```
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

ksize就是filter, 看第二个和第三个参数，表示为2X2大小。
第一个参数代表batch的数量，一般为1，第4个参数代表了channel，因为图片是有颜色channel的，一般为3个channel，因为我们这里是灰色的图片，所以这里为1。
stride和ksize是一一对应的这里的第二个参数2代表了每次在height方向上面的移动距离，第三个参数代表在width方向上面的移动距离。最后我们取出每个映射矩阵中的最大值！
####  **卷积层代码解析**
####  1、初始化

```
W_conv1 = weight_variable([5, 5, 1, 32])   5x5，patch大小，1，输入通道，32输出通道，即输出32个feature
b_conv1 = bias_variable([32]) 每一个输出通道都有一个对应的偏置量

```
#### 2、卷积
权重就是卷积核
x_image = tf.reshape(x, [-1,28,28,1])把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数

```
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) 

```
我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
 
#### **全连接层**
#### 首先设置该层的权重和偏置

```
 W_fc1 = weight_variable([7 * 7 * 64, 1024]) 输入和输出
b_fc1 = bias_variable([1024])
全连接
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) 将上一层的输出reshape成一维向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) matmul矩阵  相乘

```
#### dropout

```
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

```
tensorflow中的dropout就是：使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob

####  **输出层解析**
输入1024，输出10

```
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

```
逻辑回归，二分类问题，softmax，多分类问题 
最大化后验概率的似然函数。对似然函数（或是损失函数）求偏导
以上均为网络设置，下面才是正在训练的开始
####  **训练过程代码解析**

```
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))// 计算所有图片的交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
//最小化交叉熵
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
//判断预测值与真实值是否相同
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
//计算准确率，reduce_mean求平均值
sess.run(tf.initialize_all_variables())
//初始化所有变量
for i in range(20000):
  batch = mnist.train.next_batch(50)
  //每一步迭代，加载50个训练样本，然后执行一次train_step
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  //每一百次输出训练结果
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})//最终要的是这一步！！！！！！
  //通过feed_dict将x 和 y_张量占位符用训练训练数据替代，可以用feed_dict来替代任何张量  ，在feed_dict中加入额外的参数keep_prob来控制dropout比例

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
//:怎么知道mnist.test.image是什么
// accuracy.eval在哪里定义的
```

### 问题
softmax的具体概念？


