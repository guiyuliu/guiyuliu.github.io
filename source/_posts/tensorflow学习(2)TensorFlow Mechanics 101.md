title: tensorflow(2)Tensor Flow Mechanics101
tag: tensorflow

---
参考 [TF英文社区](https://www.tensorflow.org/get_started/mnist/mechanics)[TF中文社区](http://www.tensorfly.cn/tfdoc/tutorials/mnist_tf.html)

fully_connected_feed.py 是总体的运行过程
mnist.py中定义了四个函数，inference，training，loss，evaluation
## mnist.py
### 一、inference
就是网络结构函数，mnist.py中的inference定义的网络有一对全连接层，和一个有10个线性节点的线性层

- input:inference输入placeholder和第一层，第二层网络hidden units的个数
<!-- more -->
- 每一层都有唯一的name_scope，所有的item都创建在这个namescope下，相当于给这一层的所有item加了一个前缀
```
with tf.name_scope('hidden1'):
```

- 在每一个scope中，weight和biase由tf.Variable生成，大小根据（输入输出）的维度设置
weight=[connect from,connect to]
biase=[connect to]
- 每个变量在创建时，都会被给予一个初始化操作
```
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
```
比如weights会用tf.truncated_normal初始化器，根据给定的均值和标准差生成一个随机分布
biase根据tf.zeros保证它们的初始值都是0。

graph中主要有三个operation，两个tf.nn.relu和一个tf.matmul
最后，程序会返回包含了输出结果的logits Tensor。
### 二、loss
loss() 也是graph的一部分，输入两个参数，神经网络的分类结果和labels正确结果。进行比较，计算损失。
```
def loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')
```
这个函数分为三步

- 1.先将labels转换成所需要的格式
`tf.to_int64(labels)`这个操作可以将labels抓换成指定的格式1-hot labels，
1-hot labels：例如，如果类标识符为“3”，那么该值就会被转换为： 
`[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`
将inference的判断结果和labels进行比较
- 2.利用函数tf.nn.sparse_softmax_cross_entropy_with_logits 计算交叉熵 
- 3.计算一个batch的平均loss
```
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
```
tf.reduce_mean函数(可跨维度的计算平均值)，计算batch维度（第一维度）下交叉熵（cross entropy）的平均值，将将该值作为总损失

### 三、training 

- input: loss tensor ， learning rate
主要分为四步
- 1、 创建一个summarizer， 用来更新损失，summary的值会被写在events file里面
 `tf.summary.scalar('loss', loss)`
- 2、创建一个optimizer优化器对象tf.train.GradientDescentOptimizer（设置学习率）
- 3、创建global_step变量 ，用于记录全局训练步骤的单值
- 4、开始优化 optimizer.minimize（输入loss 和 global step）
- return train_op


### 四、evaluation
输入网络的分类结果和labels，和loss函数的输入一样
```
def evaluation(logits, labels):

  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
```

## fully_connected_feed.py
一旦图建立完之后，就可以在循环训练和评估
tensorflow/tensorflow/examples/tutorials/mnist/fully_connected_feed.py
### 总体步骤
1、设置输入
定义placeholder，函数`def placeholder_inputs(batch_size)`
2、开始训练run_rainning

- 读入数据集 
-  建立图
- 创建session
- 初始化
- 开始循环训练
  - check status
  - do evaluation

### 1.place holder
### 2.the graph
建图中所有的操作都是在`with tf.Graph（）.as_default()`下进行的
tf.graph可能会执行所有的ops，可以包含多个图，创建多个线程
我们只需要一个single graph

### 3.session
在定义完图后，需要创建一个会话session来开启这个图
- 创建session `sess=tf.session()`
- 创建initializer, `initializer=tf.global_variables_initializer`
- sess.run(initializer) 会自动初始化所有的变量


### 4.training loop
在变量初始化完成之后，就可以开始训练了
最简单的训练过程就以下两行代码
```
with step in xrange(FLAGS.max_Step)
    sess.run(train_op)
```
但是本例子要复杂一点，读入的数据每一步都要进行切分，以适应之前生成的place_holder

#### (1).fill_feed_dict
先让image_feed和labels_feed去向dataset索要下一次训练的一个batchsize的数据
```
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)
```
再讲这个数据整合成一个python字典的形式，image_placeholder 和labels_placeholder作为字典的key， image_feed和labels_feed作为字典的value
```
feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
```

#### (2).检查训练状态
see.run 在每一步训练之后都会取得两个值，loss 和train_op，（train_op不返回任何值，discard）
，所以会得到每一步的loss
每训练100次，check一下，输出loss 
每训练1000次，进行evaluation，将生成的model保存一下

#### (3).do_eval

计算整个epoch的精度
```
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
```

## 代码
### 建图步骤
- Generate placeholders for the images and labels.
- Build a Graph that computes predictions from the inference model.
- Add to the Graph the Ops for loss calculation.
- Add to the Graph the Ops that calculate and apply gradients.
- Add the Op to compare the logits to the labels during evaluation.
- Build the summary Tensor based on the TF collection of Summaries.
- Add the variable initializer Op.
- Create a saver for writing training checkpoints.
- Create a session for running Ops on the Graph.
- Instantiate a SummaryWriter to output summaries and the Graph.
- And then after everything is built:Run the Op to initialize the variables.
 Start the training loop.


### 读取数据
```
def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
```
### 开始建图
```
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
```
###开始循环训练
```
    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)

```



