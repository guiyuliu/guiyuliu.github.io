---
title: 如何实现hexo的迁移
tag: 操作
---
本文参考了知乎上两位的[回答](https://www.zhihu.com/question/21193762) [koko](https://www.zhihu.com/question/21193762/answer/138139539)

1.把之前的github.io整个repo删除掉，重新建一个
2.把这个空的repo 克隆到本地
3.将这个repo中的隐藏的.git文件夹拷贝到外面的git文件夹
``` 
cp -r .git ../git
``` 
4.本地http://CrazyMilk.github.io文件夹下通过Git bash依次执行
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

文件夹中