

title: tensorflow(3)  tf.estimator
tag： tensorflow

---
摘要:
```
classifier = tf.estimator.DNNClassifier()
train_input_fn = tf.estimator.inputs.numpy_input_fn()
classifier.train(input_fn=train_input_fn, steps=2000)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
```
---
<!-- more -->


tf.estimator 是tensorflow的一个高级的机器学习的API，使训练，evaluation 多种机器学习的模型更简单。
本文利用tf.estimator写一个深度神经网络分类器，分为以下几个步骤
1、数据预处理，讲CSV文件load进tensorflow Dataset
2、构建神经网络分类器
3、训练网络
4、evaluate
5、用写好的网络测试新的sample

### 一、数据预处理 load_csv_with_header()
该方法在learn.datasets.base里，需要三个参数

- 1.文件名，csv文件的路径
- 2.target_dtype,用numpy datatype来表示分类目标，在本例子中是花的种类0-2，是一个整数
- 3.feature_dtype，用numpy datatype来表示特征的数据类型，在本例子中是花萼的长度，宽度等等，可用float32
```
# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
```
load进来以后，dataset在tf.contrib.learn中是已命名的元组来存储的，访问方式
`training_set.data` 存储的是特征
`training_set.target` 存储的是target value
test_set同理

### 二、建立深度神经网络分类器
tf.estimator提供了一些预先定义好的模型，叫estimator,
比如可以直接用estimator来写一个DNN的网络，只需要把参数输进去，短短几行代码就能搞定

- 1、确定feature的列数，
tf.feature_column.numeric_column这个函数就是用来创建有几个feature的，在本例子中有四个feature，shape为4

```
# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir="/tmp/iris_model")
```
- 2、建立DNN分类器
直接调用，输入参数进去
这样一个有三个隐藏层，unit个数分别是10,20,10的DNN分类器就建好了

### 训练输入
数据在load进来之后，送入网络之前，还要经过input函数
tf.estimator API用input函数，作用是添加tf op来给模型提供输入数据。
```
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)
```
inputs.numpy_input_fn函数 有三个参数,数据格式是np.array
x是一个dict，键是送入DNNClassifier的feature值x，是从training_set_data里取出来的

### 三、用Iris 训练数据来训练DNNClassifier
用train方法训练，将上一步用inputs.numpy_input_fn函数加载进的数据 送入网络，设置训练步数为2000
```
# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)
```
也可以分成两个1000部来训练
如果要在训练的过程中跟踪模型，可以用SessionRunHook

### 四、evaluate the model
用evaluate方法，也是先使用input函数，再调用calssifier.evaluate方法
```
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
```

### 五、分类新的sample
用predict()方法
```
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predicted_classes))
```


### 总体流程




```flow
st=>start: CSV file
op=>operation:   load_csv_with_header()
得到training_set.data和training_set.target
op1=>operation: tf.estimator.inputs.numpy_input_fn()
op2=>operation: classifier
e=>end: output

st->op->op1->op2->e

```
