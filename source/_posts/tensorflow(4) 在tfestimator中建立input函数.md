title: tensorflow(4)在tf.estimator中建立input函数
tag: tensorflow

---
[原文链接](https://www.tensorflow.org/get_started/input_fn)
[代码链接](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/input_fn/boston.py)
上一篇我们看了使用tf.estimator直接构建一个DNN分类器，但是数据load进来之后，输入分类器之前，还要经过一个input_fn的函数。
这篇文章会教你怎么用input_fn来喂给一个神经网络回归器数据。
<!-- more -->

## 1.把feature data转换为tensor
如果你的feature/label数据是python array ，或者存在pandas dataframe 或者numpy array中，可以用下面的方法构建inputfn函数

### pass inputfn data to  your model
直接把input function作为参数输入train op，注意input fn是作为一个object 传入，而不是作为一个函数被调用，要不然会发生typeerror
即使要修改inputfn的参数，也不能这样用，有其他的方法

```
classifier.train(input_fn=my_input_fn, steps=2000) 正确
classifier.train(input_fn=my_input_fn(training_set), steps=2000) 错误
```
即inputfn在输入的时候不能调参，必须在定义的时候被设置到加载哪个数据集
若果不想重复定义，比如inputfntrain，inputfntest，inputfnevaluate
可以用以下四种方法
#### (1)、用一个包装函数 
 my_input_fn_training_set()，感觉也没简单多少
```
def my_input_fn(data_set):
  ...

def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)
```
#### (2)、创建一个固定参数的function obeject
用python的functools.partial函数创建一个function object
```
classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set),
    steps=2000)
```
#### (3)、讲input_fn包装成lambda
```
classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)
```

#### (4)、使用tf.estimator.inputs 来创键inputfn
这种方法额外的好处是可以设置其他的参数，比如设置num_epoch和shuffle来控制inputfn如何在data 上迭代。
 `tf.estimator.inputs.pandas_input_fn() `和`tf.estimator.numpy_input_fn()`
```
import pandas as pd

def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pdDataFrame(...),
      y=pd.Series(...),
      num_epochs=num_epochs,
      shuffle=shuffle)
import numpy as np

def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.numpy_input_fn(
      x={...},
      y=np.array(...),
      num_epochs=num_epochs,
      shuffle=shuffle)
```

## 2.Boston房价神经网络模型

### (1) 导入房屋数据
定义COLUMNS，FEATURES，LABEL 
定义成list，label定义成字符串

用pd.read_csv读入数据
参数意义 ?
```
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    # names:
    # return:
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
                             
```
pd.read_csv之后如下图所示
![这里写图片描述](http://img.blog.csdn.net/20170919165039685?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzYwODMzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
### (2) 定义FeatureColumns and creating the regressor
```
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
```
读入进来的feature_clos只是八个空的，如下图
![这里写图片描述](http://img.blog.csdn.net/20170919165102807?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzYwODMzNg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

要想了解一下feature_colmn对于 categorical data的用法，看后面的Linear model Tutorial
现在，创建一个DNN回归器，只需要传入两个参数，feature_columns和hidden_units
```
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],  #two hiden layer with 10 nodes each
                                      model_dir="/tmp/boston_model")
```

### (3) building the input_fn
使用了上面的第四种方法，tf.estimator.input来创建input_fn
```
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
```

利用这个函数，就可以将不同的data_set传入
另外两个参数
num_epochs: training 设为None ，这样input_fn会持续返回数据，直到达到训练步数为止
evaluation 和predict 设置为1 ，这样input_fn只迭代数据一次
shuffle: 洗牌
train：设置为 True
predict和evaluate 设置为False，所以input_fn按顺序迭代数据

### (4) train
```
regressor.train(input_fn=get_input_fn(training_set), steps=5000)
```

### (5)evaluate
```
ev = regressor.evaluate(
    input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
```
### (6)predict
```
y = regressor.predict(
    input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
# .predict() returns an iterator of dicts; convert to a list and print
# predictions
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
```
