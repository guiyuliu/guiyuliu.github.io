
title:  C++(7) 函数重载／operator
tag:  cpp

---


### （1） 函数重载
函数的名字相同，但是参数不同。可能是参数的个数不同，也可能是参数的类型不同。
<!--more -->
以下例子，print就是重载函数
```
      void print(int i) {
        cout << "Printing int: " << i << endl;
      }
      void print(double  f) {
        cout << "Printing float: " << f << endl;
      }
      void print(char* c) {
        cout << "Printing character: " << c << endl;
      }
```
### （2）运算符重载
[参考博客](http://blog.sina.com.cn/s/blog_4b3c1f950100kker.html)
operator是C++的关键字，它和运算符一起使用，表示一个运算符函数，理解时应将`operator=`整体上视为一个函数名。

在C++中，如果我们想比较两个整数，非常方便，用==就好了。但是如果我们要比较两个类对象，就不能直接用==了。要定义专门用于比较这两个类对象的`==`号
定义重载运算符函数的时候，可以定义成类成员函数，也可以全局函数。

- 定义为类成员函数
```
class person{
private:
    int age;
    public:
    person(int a){
       this->age=a;
    }
   inline bool operator == (const person &ps) const; //定义
};
```
实现方式如下：
```
inline bool person::operator==(const person &ps) const
{
     if (this->age==ps.age)
        return true;
     return false;
}
```
调用方式如下：
```
int main()
{
  person p1(10);
  person p2(20);
  if(p1==p2) cout<<”the age is equal!”< return 0;
}
```

- 定义为全局函数


