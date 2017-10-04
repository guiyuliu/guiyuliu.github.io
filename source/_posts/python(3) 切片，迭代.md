title: python(3) 切片，迭代
tag:  python

---
## 1.切片
能从list、tuple、字符串中取出指定位置的一串元素
元组是另一种list，只不过不能改变，字符串同理

从list中取出指定位置的元素很像matlab
`[a:b:c]`
<!--  more -->
a:起始序号 （第一位是0）
b:结束序号（不包括）
c:步长
也可以由负数开始取
起始序号或结束序号是最后一个，可以省略不写。
```
L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']

print('L[0:3] =', L[0:3])  
print('L[:3] =', L[:3])  从序号为0的第一个元素开始
print('L[1:3] =', L[1:3])  
print('L[-2:] =', L[-2:])  倒数第二个，包含倒数第一个
```

结果
```
lgy@lgy:~/download/python$ python slice.py 
('L[0:3] =', ['Michael', 'Sarah', 'Tracy'])
('L[:3] =', ['Michael', 'Sarah', 'Tracy'])
('L[1:3] =', ['Sarah', 'Tracy'])
('L[-2:] =', ['Bob', 'Jack'])
```

从tuple元组中取出一个切片，同理,操作的结果仍是tuple
```
>>> (1,2,3,4,5)[:3]
(1, 2, 3)

```
从字符串中取一个切片，仍是字符串
```
>>> 'ABCDEFG'[:3]
'ABC'
>>> 'ABCDEFG'[::2]
'ACEG'
```

## ２．迭代
用for循环来遍历一个list或者touple，这种循环我们称为迭代
### (1).list 迭代
```
d=['a','b','c']
for i in d:
	print i
```
输出
```
a
b
c
```
如果要给list加上下标索引，可以用`enumerate`函数将list变成索引-元素对，进行两个元素的迭代
```
 for i, value in enumerate(['A', 'B', 'C']):
    print i, value
#输出
0 A
1 B
2 C
```

### (2).字典 迭代
假设现在有一个字典d
迭代取出字典中的键值，直接在d中取
`for key in d:`
迭代取出字典中的value，for value in d.itervalue()
```
for value in d.itervalues():
	print value
# output ，no order
1
3
2
```
同时取出key-value,两个变量的迭代 for k，v in d.iteritems()
### (3).字符串迭代
```
for ch in 'ABC':
...     print ch
...
A
B
C
```


### (4).判断一个对象是否能迭代
用Iterable这个方法，`from collections import Iterable`
```
>>> from collections import Iterable
>>> isinstance('abc', Iterable) # str是否可迭代
True
>>> isinstance([1,2,3], Iterable) # list是否可迭代
True
>>> isinstance(123, Iterable) # 整数是否可迭代
False
```
### (5). 两个变量的迭代
```
for x, y in [(1, 1), (2, 4), (3, 9)]:
...     print x, y
...
1 1
2 4
3 9
```



