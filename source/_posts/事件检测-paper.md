---
title: 用少量训练数据实现单张图片的时间检测
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

### 3.3. Predicting Concept Scores for Classification 
总体的准确率要高



### 不了解的概念
训练方法较传统方法有何区别？
One shot learning？
word2vec space？
Unseen event catogories ？

