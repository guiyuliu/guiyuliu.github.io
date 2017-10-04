title:  python(5) 进程和线程
tag:  python

---
[参考](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/0013868323401155ceb3db1e2044f80b974b469eb06cb43000)
##1.进程
### 摘要
(1).有三种方法实现进程
fork() 和Process 、Pool
(2).进程间通信
Queue，Pipes
<!-- more -->
### 用三种方法实现进程
fork 
```
pid=os.fork()
```
Process
windows上没有fork，用multiprocess代替
```
from multiprocessing import Process
p=Process(target=函数名，args=(函数参数))
p.start
p.join()
```

Pool
进程池，启动大量的子进程
```
from multiprocessing import Pool
p=Pool
#调用
p.apply_async(函数名，args=函数参数)
```
进程由函数来执行
### 进程间通信
Queue
父进程创建Queue，并作为参数传给两个子进程
```
q = Queue()
pw = Process(target=write, args=(q,))
pr = Process(target=read, args=(q,))
```
## 2.线程
启动一个线程就是把一个函数传入并创建Thread实例，然后调用start()开始执行：
```
t = threading.Thread(target=函数名, name='线程名')
t.start()
t.join()
```

###　线程锁
多进程中，同一个变量都会拷贝一份存在于各个进程中。但是多个线程中，变量由所有线程共享，当多个线程同时改一个变量时，容易把变量给改乱了。

高级语言的一条语句在CPU执行时是若干条语句，即使一个简单的计算：
`balance = balance + n`
也分两步：
1.计算balance + n，存入临时变量中；
2.将临时变量的值赋给balance。
第一个线程只执行完第一步，就执行第二个线程，这样变量还没赋值完成，就又被改变，变量就改乱了。

用法:
```
def run_thread(n):
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            # 放心地改吧:
            change_it(n)
        finally:
            # 改完了一定要释放锁:
            lock.release()

```

在python上加线程锁没有什么意义，因为python有个GIL全局锁，即使是100核的CPU，每次也只能执行一个线程。CPU永远跑不满。
而C,C++,JAVA等高级语言就不一样了。四核的CPU，写四个死循环，四个CPU的占用率都会达到100%

一个线程只有一个锁
由于可以存在多个线程锁，不同的线程有不同的锁，试图获得对方持有的锁时，可能会造成死锁

### ThreadLocal
局部变量只有自己的线程能看见，不会影响其他线程，而全局变量要加锁。
如果普通的全局变量，设成一个dict，不同的线程所需要的对应不同的key，这样访问时会比较麻烦。
可以创建一个threadlocal的对象。
`local_school = threading.local()`
这个对象可以看成一个dict，只是不同的线程访问时不需要制定是哪个线程，它会自动处理。
