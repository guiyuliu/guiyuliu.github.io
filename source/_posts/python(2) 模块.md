
title: python(2) 模块
tag: python

---
### 模块的package组织
python模块可以避免不同模块中，相同名字的函数，相同名字的变量之间发生冲突
如何避免？
用包package来组织，包就相当于给模块名加了一个前缀，这样能避免冲突。
<!-- more -->
>注意，每个包的文件夹里必须有一个__init__.py的文件，即使init是空的也要有这个文件！这个文件是识别package的标志

一个package下面可以有子文件夹，子文件夹里也要有__init__.py文件

比如一个包的名字叫bao，包里有个模块叫xyz.py,引入了包之后这个模块的名字变成了bao.xyz
如果还有下一级的子文件夹abc，子文件夹里有个wq.py，那这个模块叫bao.abc.wq

### hello.py模块
```
#!/usr/bin/env python
# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Michael Liao'

import sys

def test():
    args = sys.argv
    if len(args)==1:
        print 'Hello, world!'
    elif len(args)==2:
        print 'Hello, %s!' % args[1]
    else:
        print 'Too many arguments!'

if __name__=='__main__':
    test()
```
#### 注释
一般模块的前两行是注释
```
#!usr/bin/env python
# -*- coding:utf-8 -*-
'a test moudle'
```
第三行为注释，任何模块的第一个字符串都被认为是文档注释

#### sys 
`import sys`
sys.argv是一个存储命令行参数的变量，存为list
sys.argv[0]就是模块名
比如运行python hello.py michael 获得的sys.argv就是['hello.py','michael']

####  `__name__=__main__`
```
if __name__='__main__'
    test()
```
name是python一个内置的特殊变量，它用来判断当前模块是被运行了，还是被调用。
如果是被运行，则if条件成立，运行test()函数。如果是被当做模块输入 import hello ，则if条件不成立,要想运行test，必须hello.test来调用。

### 变量函数的作用域
python中的变量或者函数大概有三种命名方式
|命名方式 | 示例 | 类型 |
| ----| ----: | :----:|
|正常的 |flow ，x ，y|公开的public|
|前后都有两个下划线的|`__name__` `__doc__`|特殊变量|
|前面有一个或者两个单划线的|'_a',`__aa`|私有变量（private）|

一般私有变量或者函数都是不希望被引用的，虽然不能限制你引用。比如私有的函数是为了更好的封装代码。

### python模块的搜索路径
搜索顺序： 当前目录，所有已安装的内置模块，安装的第三方的库，
搜索路径在sys.path这个变量中
```
import sys
sys.path
```

有两种方法添加模块的搜索路径
1.直接修改sys.path,不过这种方法只在文件运行时有效，运行结束后失效
```
import sys
sys.path.append('/usr/local/.....')
```
2.修改PYTHONPATH，修改方法和修改Path环境变量类似。注意只要添加你自己的搜索路径，python本身的搜索路径不受影响。

### `__future__模块`
如果想在python2.x中使用python3.x的功能，要导入`__future__`模块
python2.x和python3.x的区别
比如除法`/`在2.x中代表地板除。在3.x中`//`才代表地板除，去余数取整。
如果想在python2.x中运用python3.x的除法，要导入
`from __future__ import division`

还有字符串的表示方法不同，为了适应python3.x的新的字符串表示方法，在2.7的版本中，可以通过导入
`from __future__ import unicode_literals`