# Programming Massively Parallel Processors
## 介绍
此项目为学习《Programming Massively Parallel Processors》时的代码记录，主要是第7~12章的并行计算模式代码。
主要包含Convolution, Prefix Sum, Histogram Computation, Sparse Matrix Computation, Merge Sort, Graph Search 6大计算模式。
由于书中没有完整代码，所以此项目中代码并不一定是最优解，只是按照书中讲解自我推理得来。
<br>
## 简单笔记
### Convolution
* 卷积核体积小，初始化后不会变，适合使用__constant__保存
* 图像体积大，需要使用tile策略，同一个像素点可能被block中多个thread使用，适合读入__shared__中
* 图像数据读入__shared__中时，尽量保证coalescing的方式读入
* 对于halo cells可以不用特殊处理，因为其他block在将其载入__shared__中时L2 Cache会将其缓存起来，
### Prefix Sum
* 浮点数计算存在大数吃小数的情况，可以使用Kahan算法进行一定程度的补偿
* 按照Brent Kung和Kogge Stone累加算法，读取数据效率不高，故采用corner tuning的方式，将数据先以coalescing的方式载入__shared__
* 通常情况下Kogge Stone算法直观性更好，而Brent Kung则是在速度和前者差不多的情况下功耗更低
* 对于任意长度的累加序列，可以采用三段法。但是由于要多次读写全局内存，效率十分底下，可以采用Single-Pass的方式。
起难点在于需要实现block之间的通信和同步，采用全局内存+原子操作解决。
### Histogram
* 直方图统计需要用到atomicAdd原子操作，而原子操作是否受限于内存读取速率，因为同一个内存地址同时只能进行一个原子操作，效率十分低下。而Cache也能一定程度的缓解这个问题
* 对于超大序列的统计需要用到多个block并行，而太多的请求即使在Cache中也难以解决，故考虑私有化，每个block中先用__shared__进行统计，最后再汇总
* 对于某些统计，例如图像直方图统计中，存在多个连续相同地址的atomicAdd操作请求，故考虑聚合，将多次请求合并为一次，提升效率
* 书中关于Aggregation的代码Figure.9.11有错误，每次累加的位置应当是prev_index坐标的地址。