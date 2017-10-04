
title:  python(4) list生成式
tag:  python

---
要生成一个列表list，如果这个list有一定的规律，可以用循环生成。python的内置函数range()可以直接生成list
如下
```
>>> range(1, 11)
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```
<!-- more -->
在matlab里面，如果要生成这样一个矩阵，首先定义一个空矩阵，然后通过for循环不断往里push值，在python里面也可以这样操作，但是代码写起来太麻烦了，比如说我们要生成一个list
`[1x1, 2x2, 3x3, ..., 10x10]`
用for循环代码如下
```
L=[]
for x in range(1,11)
    L.append(x*x)
```
其实在python里面，不用for循环，一行列表生成式就可以达到上面的目的
### 一层循环
```
>>> [x*x for x in range(1,11)]
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```
把要生成的元素x * x放到前面，后面跟for循环，就可以把list创建出来

### 两层循环
```
>>> [m + n for m in 'ABC' for n in 'XYZ']
['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']
```
由结果可以看出:两个字符串相加是字符串

### 两个变量的循环
其实在之前字典里，利用dict.iteritem()就可以同时循环键和键值
```
>>> d = {'x': 'A', 'y': 'B', 'z': 'C' }
>>> for k, v in d.iteritems():
...     print k, '=', v
... 
y = B
x = A
z = C
```
现在列表生成器也可以用两个变量来生成list
```
 d = {'x': 'A', 'y': 'B', 'z': 'C' }
[k+'='+v for k,v in d.iteriterms()]
['y=B', 'x=A', 'z=C']
```
 
 最后把一个list中所有的字符串变成小写：
```
>>> L = ['Hello', 'World', 'IBM', 'Apple']
>>> [s.lower() for s in L]
['hello', 'world', 'ibm', 'apple']
```
s指代列表中的所有字符串，s.lower()方法是把大写字母变成小写，是字符串的方法，如果有整数在里面，就会报错
三层以上循环就很少用了




