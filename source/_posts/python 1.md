title: python(1) python基础知识

tag: python

---
## 基础
### 注意事项
python编辑器推荐notepad++或者sublime，windows下不推荐word或者自带文本编辑器，因为word存储的不是文本格式，而文本编辑器会自作聪明的在前面加上UTF8-BOM

### 输入输出
输出 print
输入 python提供了`a=raw_input()`函数将输入存为字符串，保存进一个变量
<!-- more -->
raw_input读入的数字也会是字符串的形式，如果读入一个整数，要先把字符串用int()转换成整数， a=int(raw_input('input an integer:'))

### 字符串和编码
存储的时候会用utf-8变长码方式编码，而在内存中是用Unicode的方式编码

### 列表list和元组tuple
列表用方括号，元组用圆括号
获取列表长度 `len(list)`
列表从0开始计数
删除元素`list.pop(i) `
在后面增加元素`list.append()`
在指定序号增加元素`list.insert(2,'a')`
将某一个元素赋值为另一个值`list[2]='y'`
list里的元素可以是数字、字符串、布尔，也可以是list,如下所示
```
list=['aa',123,True,['xx','yy']]
list[3][1]='yy'
```
获取list的最后一个元素是list[-1]

元组
元组的元素不能增、删、改
如果元组里的元素有一个list，那么我们可以改变list里的值，为什么这里又能变呢，因为元组指向的list是不变的
创建只有一个元素的元组时，必须要加逗号` t=(1,)`,如果写`t=(1)`,python就会理解成`t=1`

### 条件判断
elif是else if的缩写，条件判断是从上往下判断，只要碰到正确的了，剩下的条件就不再判断了。

### 循环
for .. in .. 循环
for x in  ，就是把list/tuple中的每一个元素代入进x，然后执行语句

while循环
一般会放在循环体内部自减，自增。

###  dict字典
很好理解，有一个索引key，索引里对应着一个值value，用大括号括起来
```
 d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
```
删除里面的元素也是dict.pop(''),直接删除key，会把对应的value也删除掉
赋值的时候dict['x']=11

dict一个最大的特点就是方便查找，查找的时候有两种方法，一种是in，一种是get
'Thomas' in dict
如果存在就会返回对应的值,不存在就返回False

dict.get('Thomas')
如果存在就会返回对应的值，如果不存在就返回None，None在命令行里面什么都不显示

set 和dict差不多，但是只存储键，不能存储value
创建一个set的时候要传入一个list 
```
a=set([1,2,3,4])
a
{1,2,3}
```

## 函数
### 递归函数
在函数内部可以调用其他函数，在函数内部调用自己的，称为递归函数。最简单的递归函数就是N的阶乘fact(n)=n*fact(n-1)

因为函数每调用一次，在内存中栈就会增加一层，每返回一次，就会减少一层，所以当递归次数增多时，可能会导致栈溢出









## 其他
### with
```
with  xxx:
     with-body
```

上下文管理器 ,执行的时候调用enter ，执行完with-body 之后调用 exit

### 
`sys.modules`是一个字典，它包含了从 Python 开始运行起，被导入的所有模块。


`"I'm %s. I'm %d year old"` 为我们的模板。%s为第一个格式符，表示一个字符串。%d为第二个格式符，表示一个整数。('Vamei', 99)的两个元素'Vamei'和99为替换%s和%d的真实值。 

`assert condition`
trigger an error if the condition is false.

 `_variable_with_weight_decay`变量函数前加下划线
单下划线开头，这个被常用于模块中，在一个模块中以单下划线开头的变量和函数被默认当作内部函数，如果使用 from a_module import * 导入时，这部分变量和函数不会被导入。不过值得注意的是，如果使用 import a_module 

Python主要有三种数据类型：字典、列表、元组。其分别由花括号，中括号，小括号表示。
如：
`字典：dic={'a':12,'b':34}`
字典的每个键值(key=>value)对用冒号(:)分割，每个对之间用逗号(,)分割，整个字典包括在花括号({})中 ,格式如下所示：
d = {key1 : value1, key2 : value2 }
列表：`list=[1,2,3,4]`
元组：`tup=(1,2,3,4)`






