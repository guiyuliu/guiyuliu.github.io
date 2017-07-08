---
title: 如何实现hexo的迁移
tag: 操作
---
本文参考了知乎上两位的[CrazyMilk](https://www.zhihu.com/question/21193762/answer/79109280) [koko](https://www.zhihu.com/question/21193762/answer/138139539)
### 一、设置github.io的两个分支
总体思路：一个分支用来存放Hexo生成的网站原始的文件，另一个分支用来存放生成的静态网页。
<!-- more -->
1.把之前的github.io整个repo删除掉，重新建一个
2.把这个空的repo 克隆到本地
3.将这个repo中的隐藏的.git文件夹拷贝到外面的git文件夹
``` 
cp -r .git ../git
``` 
4.本地github.io文件夹下通过Git bash依次执行
``` 
npm install hexo
hexo init
npm install 
npm install hexo-deployer-git
``` 
5.把.git文件重新拷贝进来,之所以要先拷出去是因为执行hexo init 要求文件夹必须为空

6.新建一个branch hexo,这个用来存本地文件，写一个readme.md push上去，要不然是branch是空的，查看自己在哪个分支时没法看
``` 
git branch -b hexo
git add readme.md
git commit -m ”test branch“
``` 
再新建一个master的branch

7.  修改_config.yml中的deploy参数，分支应为master,依次执行
``` 
git add .
git commit -m "..."
git push origin hexo
``` 
提交网站相关的文件；
8. 执行hexo g -d生成网站并部署到GitHub上

### 二、关于日常的改动流程在本地对博客进行修改（添加新博文、修改样式等等）
通过下面的流程进行管理。
1. 依次执行
``` 
git add .
git commit -m "..."
git push origin hexo
``` 
指令将改动推送到GitHub（此时当前分支应为hexo）；
2. 然后才执行hexo g -d发布网站到master分支上。

### 三、本地资料丢失后的流程
1. 使用
``` 
git clone https://github.com/guiyuliu/guiyuliu.github.io.git
``` 
拷贝仓库（默认分支为hexo）
2. 在本地新拷贝的github.io文件夹下通过Git bash依次执行下列指令：
``` 
npm install hexo
npm install
npm install hexo-deployer-git
``` 
（记得，不需要hexo init这条指令）。




