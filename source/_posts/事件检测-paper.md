---
title: 用少量训练数据实现单张图片的事件检测
tag: PaperReading
---
### 本主要贡献：
1.从网络数据发现事件相关的概念，删除某些杂乱的概念
2.一个concept-based表示方法
3.RED包含稀少的事件，21个种类，7000张图片，更有挑战性。
<!-- more -->
## Approaches
### 3.1. Event Concept Discovery
为图片生成一个简短的表示 

- 从维基上挑选150 generic social events
每个events从flicker上挑选200张图片 收集这些图片的tag ，tag的分割

- 通过谷歌的数据集，为每个event选择20个最近的neighbor

- 将谷歌和flicker的segments结合

最终有856个event concepts,concept不仅包含物体，场景和动作，也包含下属的events和他们的类型
### 3.2. Training Concept Classifiers
比起直接用深度CNN feature把图片分到对应的events类中用有限的训练数据，我们的方法要好。
给一个concepts的集合，每个concept，从维基上找到前100个图片，但是相似的concept，检索出来的图片会重合
解决：对所有的concepts进行聚类，用minibatch k最近邻法
对第i个concept 我们获得了y-size的聚类
这样聚类是为了挑选训练图片
### 3.3. Predicting Concept Scores for Classification 
总体的准确率要高



### 不了解的概念
训练方法较传统方法有何区别？
One shot learning？
简而言之就是用少量样本训练网络
word2vec space？
Unseen event catogories ？
the stickiness of a segment

扩展阅读
transfer learning/zero shot learning 参考[罗浩.ZJU](https://www.zhihu.com/question/53794313/answer/136861035)的回答
zsl 就是训练两个网络，一个是输入到特征描述空间的网络，一个是特征描述空间到输出的网络。
比如说现在要进行三种语言之间的翻译，按照传统的方法需要训练至少六个网络，双向的话需要12个网络，但是现在我没有那么多样本，我只需要训练
- 英语→特征空间→日语
- 韩语→特征空间→英语
这两个网络，那么就可以自动学会
- 韩语→特征空间→日语
这个翻译过程，我用许多小的样本就可以学习出一个能力非常强大的网络，我甚至都没有韩语到日语这个样本我就学会了这个过程，这就是zero shot learning
one shot learning的zero shot learning类似，只不过提供少量或一个样本
