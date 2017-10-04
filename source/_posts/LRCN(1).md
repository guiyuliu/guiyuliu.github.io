title: LRCN(1)
tag: 代码复现

这篇文章描述了如何复现《Long-term Recurrent Convolutional Networks for Visual Recognition and Description》这篇paper的动作识别的实验部分
<!--more-->
这里是复现代码的指导[原文链接](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video])
当然，重新复现别人的代码总会遇到各种各样的问题。我把他的代码使用过程翻译一遍，然后按照自己的复现过程整理各种各样的问题，供大家参考。


## 一、我的使用过程
### １、从github上克隆整个仓库
打开命令行，输入
```
git clone https://github.com/LisaAnne/lisa-caffe-public.git
```
克隆完之后你会发现这个仓库跟caffe的文件夹结构一样，没错，这就是caffe,只不过是作者升级过的caffe,你会发现在example这个文件夹里多出了两个例子/LRCN_activity_recognition和LRCN_evaluate。当然还做了一些其他的修改，这就不细说了。所有关于action recognition的代码都在examples/LRCN_activity_recognition这个文件夹中。

### ２、重新编译caffe
我想大家已经编译过无数次caffe了，如果你已经安装过官网给的caffe，不用管它，这里还是要重新编译一遍。在重新编译之前，需要修改你的环境变量，即在~/.bashrc中把之前的PYTHON_PATH给注释掉，换成现在的这个文件夹路径
sudo gedit ~/.bashrc
比如我安装官网的caffe时加入了

```
＃　export PYTHONPATH=/home/lgy/software/caffe/python:$PYTHONPATH
```
现在我把它注释掉（前面加＃号），并换成这个caffe仓库的路径

```
export PYTHONPATH=/home/lgy/workspace/lisa-caffe-public/python:$PYTHONPATH
```
然后就可以开始编译这个仓库了，关于Makefile.config怎么修改，就按照你之前怎么修改的做好了,不过它好像不支持cudnn.我就没有注释cudnn。还有作者说的要保证设置WITH_PYTHON_LAYER := 1
编译的过程就是在根目录下运行以下四行代码

```
make all -j4
make test
make runtest
make pycaffe
```
编译成功啦！
### ３、提取RGB frame
顾名思义，就是把视频变成一帧帧的图片。进入直接运行extract_frames.sh，在该脚本文件中修改要提取的视频的路径，参数是frames/s

```
cd /home/lgy/workspace/lisa-caffe-public/examples/LRCN_activity_recognition
./extract_frames.sh video.avi 30/1
```

### ４、提取光流
如果你在提取光流的过程中出现关于mex_OF的什么错误，不要着急，你还需要下载一个[][代码](https://lmb.informatik.uni-freiburg.de/resources/software.php)。
在该页面中找到High accuracy optical flow estimation based on a theory for warping这篇文章的Download Matlab Mex-functions for 64-bit Linux, 32-bit and 64-bit Windows，下载之．
所需文件 create_flow_images_LRCN.m、把这里面的两个函数分开来存放，要不然找不到下面的一个函数 


###  ５、下载模型
有四个个模型，single frame model的两个(RGB和flow)和LRCN model的两个，这里是作者训练好的模型。[模型链接](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video_weights.html)把这四个模型都下载下来。模型还蛮大的呢，每个都有200多Ｍ。
如果你想自己重新训练的话，按照作者说的。single frame model是在这个模型上finetune的，LRCN model是用single frame model训练的。具体的过程我还没试过。
### ６、运行demo
下载好的模型直接放在LRCN_activity_recognition这个文件夹里
frames和flow_images分别存放图片和光流图片，按照代码，这两个下属文件夹下面应该还设置小的文件夹，文件夹名称即为你要测试的video的名称



##  二、作者原文
### Code:
 All code to train the activity recognition models is on the "lstm_video_deploy" branch of Lisa Anne Hendricks's Caffe fork. All code needed to replicate experiments can be found in "examples/LRCN_activity_recognition". 
所有用来训练动作识别模型的代码在这个仓库[lisa-caffe-public](https://github.com/LisaAnne/lisa-caffe-public)的lstm_video_deploy这个分支上.
用来复现实验的代码在examples/LRCN_activity_recognition这个文件夹中。

### Data:
 The model was trained on the UCF-101 dataset . Flow was computed using [1]. 
该模型是在UCF-101这个数据集上训练的
###Models:
 Single frame and LRCN models can be found here. 
模型链接

**NOTE** 
Some people have had difficulty reproducing my results by extracting their own frames. I am almost positive the issue is in the ``extract_frames.sh'' script, but have not had time to track it down yet. You can find the RGB and flow frames I extracted here. 
如果在提取视频帧和光流上有困难，你可以从这里下载我提取好的ＲＧＢ和光流帧

Steps to retrain the LRCN activity recognition models: 
1. Extract RGB frames: The script "extract_frames.sh" will convert UCF-101 .avi files to .jpg images. I extracted frames at 30 frames/second. 
2. Compute flow frames: After downloading the code from [1], you can use "create_flow_images_LRCN.m" to compute flow frames. Example flow images for the video "YoYo_g25_c03" are here. 
3. Train single frame models: Finetune the hybrid model (found here) with video frames to train a single frame model. Use "run_singleFrame_RGB.sh" and "run_singleFrame_flow.sh" to train the RGB and flow models respectively. Make sure to change the "root_folder" param in "train_test_singleFrame_RGB.prototxt" and "train_test_singleFrame_flow.prototxt" as needed. The single frame models I trained can be found here. 
　　训练单帧模型，用这连个脚本分别训练RGB和光流的模型。确保修改train_test_singleFrame_RGB.prototxt和train_test_singleFrame_flow.prototxt中根目录的路径。
4. Train LRCN models: Using the single frame models as a starting point, train the LRCN models by running "run_lstm_RGB.sh" and "run_lstm_flow.sh". The data layer for the LRCN model is a python layer ("sequence_input_layer.py"). Make sure to set "WITH_PYTHON_LAYER := 1" in Makefile.config. Change the paths "flow_frames" and "RGB_frames" in "sequence_input_layer.py" as needed. The models I trained can be found here. 
　　训练LRCNmodel 用单帧模型作为起点，通过运行"run_lstm_RGB.sh" "run_lstm_flow.sh"这两个脚本来训练LRCN模型
　　LRCN模型的数据层是python层，"sequence_input_layer.py"。保证编译caffe时设置WITH_PYTHON_LAYER := 1。修改sequence_input_layer.py脚本中 "flow_frames" 和 "RGB_frames"的路径。
5. Evaluate the models: "classify.py" shows how to classify a video using the single frame and LRCN models. Make sure to adjust the pathnames "RGB_video_path" and "flow_video_path" as needed. You can also evaluate the LSTM model by running code found in "LRCN_evaluate" (added 1/12/16). 
　　评估模型：classify.py展示了如何用single frame model和LRCN model分类一个video。如果需要的话，修改其中"RGB_video_path" 和 "flow_video_path"的路径。
用LRCN_evaluate中的内容来评估LSTM模型

