# ParaClickHouse
A very crude implementation of CUDA Heterogeneous ClickHouse Aggregate Functions.



## 概述

- 本项目尝试将ClickHouse聚合函数的计算进行CUDA异构加速
- 硬件：Intel CORE i5; GeForce RTX 3060
- 操作系统：Ubuntu22.04.2
- ClickHouse版本：ClickHouse 22.12.1.1
- 编译器版本：LLVM/Clang 15.0.0
- CMake版本：3.24.2
- CUDA Driver/Runtime版本：12.1/12.1
- 项目展示了在ClickHouse源码基础上作了改动的目录及文件。parallelization文件夹是新增的，其余改动都标记了"kai mod"注释或"ifdef ENABLE_CUDA"宏;



## 核心思想

- 多线程并发启动网格级核函数并行，通过异步流形成Cpy-Kernel-Cpy(CKC)三级流水。
- 在ClickHouse解析器确定有Sum聚合查询时，预分配主机端内存空间/设备端内存空间/流，以达到复用效果。
- 扩大了ClickHouse批处理大小，以掩盖PCI-e传输的额外开销。
- 因为没有利用锁页内存，所以该异构算法受限于H2D传输。
- 在Star Schema Benchmark的lineorder表上，聚合480,003,275行数据可以达到1.14～1.66倍提速。



## 限制

- 只能作 “select sum(filed) from table;"操作。



> 本人C++/CUDA/CMake/大型项目的开发经验为零，项目阅览可以权当图一乐。
