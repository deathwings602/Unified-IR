# Introduction To Unified IR

## Overview

**Unified IR** 是一个描述张量程序的统一中间表示，旨在为人工智能应用提供跨平台的统一编译优化。每个张量程序都对应于 **Unified IR** 中的一个 **Kernel Graph**，描述了不同的设备算子在张量程序中的数据依赖关系；**Kernel Graph** 中部分节点为自定义算子，表示为一个 **iGraph**，这是一个对描述算子实现的指令级中间表示。

## Kernel Graph

在 **Unified IR** 中，**Kernel Graph** 用于描述一个张量程序，其中的每个节点表示一个在目标硬件平台上执行的算子，每条边表示在不同算子之间共享的张量。所有的张量都存储在设备的DRAM中，这是因为不同的算子无法通过寄存器文件或者SRAM共享数据。每个节点所表示的算子可以是对应硬件预定义的算子，如矩阵乘法和卷积，也可以是由 **iGraph** 描述的自定义算子，这类算子允许我们在 **Kernel Graph** 层级上做算子融合等算子间（inter-operator）优化。

## iGraph

**iGraph** 从硬件指令的角度描述了算子的实现，它被组织成一个数据流图的形式，图中的每个节点为 **iOp**，表示在硬件的最小并行单元上的一条硬件指令或是一个指令序列，每条边是一个 **iSlice**，表示上述指令所处理的数据切片。为了表示硬件的不同处理单元如何协同，**iGraph** 还包含了硬件并行架构和算子实现并行方式的信息。

### Hardware Model

考虑到AI加速硬件内在的相似性，**Unified IR** 提出了一套高层次的对硬件并行架构的建模，从而可以用一组简单的参数描述不同的硬件的特性。

我们定义 **最小并行单元** 为目标硬件中可以独立执行的最小单元，这在不同的硬件中有不同的含义，在NVIDIA系列的GPU上对应于SM（Streaming Multprocessor）上的一个调度器（独立的调度CUDA程序中的一个Warp），而在Ascend 910系列芯片中则对应于一个AI Core。在部分硬件中，多个最小并行单元可以在一个算子中协同进行计算，它们之间可以通过SRAM和组内同步来进行通信，我们称这些最小并行单元形成的结构为 **并行组**。例如在CUDA编程中，一个线程块包含多个Warp，它们都在同一个SM上执行，彼此之间可以进行同步和通过共享内存同步数据。多个 **并行组** 最终组成整个AI加速硬件。

我们按照并行架构对内存层级也进行了类似的划分，从高到低分为 **REG**、**SRAM** 和 **DRAM**。**REG** 层级的内存由每个最小并行单元独占，访问速度最快；**SRAM** 层级的内存则可以在不同的并行组之间共享，访问速度次之；**DRAM** 在不同的算子之间共享，访问速度最慢。

因此，根据上述的 “硬件 -> 并行组 -> 最小并行单元” 的抽象，我们可以用二元组 (numGroup, numUnit) 来描述硬件的并行架构，分别表示一个硬件中有多少个并行组，一个并行组中有多少个最小并行单元。由于CUDA的编程接口对底层的硬件架构做了进一步的抽象，一个算子可以启动的线程块数量可变，每个线程块中的Warp数量也可变，因此我们这套模型在描述CUDA算子时有一些微妙的调整，我们令 numGroup 对应于算子实现中的线程块数量，而 numUnit 对应于线程块中的Warp数量。此外，Ascend 910 系列芯片的AI Core之间不能进行通信，因此就不存在上述的 **并行组** 层级，我们可以将 numUnit 设置为1来进行描述。

类似的，我们也可以将具体硬件的内存层级按照其和并行架构的关系映射到上述的 “REG -> SRAM -> DRAM” 层级上。对于NVIDIA的GPU来说，寄存器文件对应 **REG**，共享内存对应 **SRAM**，全局内存对应 **DRAM**。而对于 numUnit = 1，即不存在 **并行组** 层级的硬件，其没有对应于 **SRAM** 的内存层级。

### Parallel Description

**iGraph** 中的并行方式描述包含两个维度，分别是并行维度和循环维度，用元组 (numParallel, numLoop) 表示。前者表示算子实现中将计算任务在 numParallel 个最小并行单元中并行执行，并为每一个最小并行单元分配一个取值于 [0, numParallel) 的和硬件并行架构相容的并行编号。例如，对于硬件架构由 (108, 4) 来描述的硬件，其 numParallel = $108 \times 4$ = $432$，第1个并行组中的4个最小并行单元的编号为0、1、2、3，第2个并行组为4、5、6、7，以此类推。编号用于为不同的单元分配计算任务，其相容性让我们可以在并行组的层级上对数据的复用进行优化。后者表示了每个并行处理单元内会做长度为numLoop的串行循环，执行由 **iGraph** 内的 **iOp** 所定义的指令序列。其语义如下：

```python
for parallel_id in 0...numParallel: # Parallel
    for loop_id in 0...numLoop: # Series
        loopBody(parallel_id, loop_id)
```

### iSlice

**iSlice** 用于表示每个并行处理单元在一次循环中所需要处理的数据切片。它表示为一个对底层的 **iPointer** 的wrapper，**iPointer** 表示在某个层级上的一块连续的内存，记录了内存层级、数据类型、首地址和内存大小等信息。其中，`dram` 上的iPointer在全局唯一，`sram` 上的iPointer在每个并行组内唯一，而 `reg` 上的iPointer则为每个最小并行单元独占。**iSlice** 则在 **iPointer** 的基础上表示其所指向的内存的一个二维的切片，由一组连续的内存片段组成，具有属性 `Shape`、`Stride` 和 `Offset`。`Shape` 是一个二元组，表示该内存切片的形状，`Stride` 也是一个二元组，表示在不同维度上的步长，`Offset` 则是一个关于 `parallel_id` 和 `loop_id` 的仿射函数，用于表示每一个最小并行单元在每个循环中所处理的内存切片的的首地址相对于其所基于的iPointer的偏移，用于表示输入输出数据和中间变量在不同并行单元和循环上的划分。

```
iPointer ::= iPointer(ID, Hierarchy, DType, Size)
ID ::= Int
Hierarchy ::= reg | sram | dram
DType ::= fp64 | fp32 | fp16 ...
Size ::= Int

iSlice ::= iSlice(iPointer, Offset, Shape, Stride)
Offset ::= Affine Function of (parallel_id, loop_id)
Shape ::= (Int, Int)
Stride ::= (Int, Int)
```

### iOp

基于上述的 **iSlice** ，我们进一步定义在 **iSlice** 上的指令级操作 **iOp**。作为对硬件指令的抽象，一个 **iOp** 通常可以在一个最小并行单元上由一条硬件指令或者一个硬件指令序列完成。**iOp** 分为访存iOp和计算iOp，分别用来表示访存操作和计算。

#### 访存iOp
访存iOp分为两种，一种是移动（Move），另一种是同步（Sync）。

移动iOp表示在不同的内存层级间移动内存切片，或者在同一内存层级内对内存切片的布局进行改变，对应硬件指令上的内存相关指令，其语义是将位于 `from` 上的内存切片 `s1` 搬运到位于 `to` 上的内存切片 `t1` 上，并且搬运的数据类型为 `dtype`。实际上，由于 `s1` 和 `t1` 作为内存切片不仅包含其基于的指针、地址偏移、形状和步长等信息，也包含了该内存切片所在层级和内部的数据类型，因此 `move` 指令中的属性均可以从输入输出中推导出来，这里将其显式定义出来是为了易读性。如无特殊说明，下面介绍的其他 **iOp** 定义也遵循这一约定。

```c
move.from.to.dtype t1, s1

.from = {.dram, .sram, .reg}
.to = {.dram, .sram, .reg}
.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}

// examples
move.dram.sram.fp32 t1, s1
move.sram.reg.fp16 t1, s1
```

同步iOp则表示需要在对应的内存层级上对相应的并行结构做同步，从而保证数据依赖的正确性。可以注意到 `sync` 指令要求 `Pointer(t1) = Pointer(s1)`，这是因为只有当算子实现中写入和读出的内存是同一块时才需要进行同步。例如，`s1` 和 `t1` 都对应于同一个 `dram` 上的指针所指向的空间，也有相同的形状和步长，但是二者的地址偏移不同，这就意味着一个最小并行单元所读取的切片可能是由别的单元写入的，因此需要进行同步才能保证数据依赖的正确性。

```c
sync.scope t1, s1

.scope = {.dram, .sram, .reg}
s.t. Pointer(t1) = Pointer(s1)

// examples
sync.sram t1, s1
```

#### 计算iOp

计算iOp有几种类型，有 `unary`、`binary`、`broadcast`、`reduce` 和 `identity`。`unary` 为单目运算，语法如下，其中 `name` 表示运算类型，`dtype` 表示数据类型，`params` 为可选参数，用于 `muls`、`leaky_relu` 等指令。

```c
unary.name.dtype t1, s1, [params]

.name = {.muls, .adds, .divs, .subs, .exp, .log, .sin, .cos, .relu, ...}
.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}

// examples
unary.muls.fp32 t1, s1 [2.0f, ]
unary.exp.fp32 t1, s1
```

`binary` 为双目运算，其定义和 `unary` 类似。

```c
binary.name.dtype t1, s1, s2

.name = {.mul, .add, .div, .sub, ...}
.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}

// examples
binary.mul.fp32 t1, s1, s2
binary.sub.fp32 t1, s1, s2
```

`broadcast` 和 `reduce` 表示广播和规约iOp，其中 `dtype` 表示数据类型，`axis` 表示在进行规约或者广播的轴。而 `scope` 表示在什么范围内进行规约或者广播，其中 `unit` 表示在每个最小并行单元内，而 `group` 表示在同一并行组内的多个最小并行单元之间，`op` 表示用于规约的计算类型。对于 `group` 层级上的规约或者广播涉及到不同并行单元，因此需要在 `sram` 上通信，所以需要在 `params` 中指定一个位于 `sram` 上的iSlice作为buffer来辅助，另外参数groupSize指定的进行广播或者规约的组的大小，需要满足 `numUnit % groupSize == 0`。

```c
broadcast.axis.scope.dtype t1, s1, [params]

.axis = {.col, .row}
.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}
.scope = {.unit, .group}

// example
broadcast.row.unit.fp32 t1, s1 // Shape(s1) = (4, 1), Shape(t1) = (4, 128)
broadcast.col.unit.fp32 t1, s1 // Shape(s1) = (1, 128), Shape(t1) = (4, 128)

broadcast.row.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 1), Shape(t1) = (4, 128), numUnit = 8
broadcast.col.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (1, 128), Shape(t1) = (4, 128), numUnit = 8
```

对于 `group` 层级上的规约，其最终规约的结果会保存在 `parallel_id % groupSize == 0` 的最小并行单元上；而对于 `group` 层级上的广播，其广播的值来自 `parallel_id % groupSize == 0` 的最小并行单元。

```c
reduce.op.axis.scope.dtype t1, s1, [params]

.op = {.max, .min, .add, .mul}
.axis = {.col, .row}
.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}
.scope = {.unit, .group}

// examples
reduce.add.row.unit.fp32 t1, s1 // Shape(s1) = (4, 128), Shape(t1) = (4, 1)
reduce.add.col.unit.fp32 t1, s1 // Shape(s1) = (4, 128), Shape(t1) = (1, 128)

reduce.add.row.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 128), Shape(t1) = (1, 128), numUnit = 8
reduce.add.col.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 128), Shape(t1) = (1, 128), numUnit = 8
```

除了以上这些iOp之外，还有一类不表示具体计算但是对Unified IR的表达性十分重要的iOp，称为 `identity`。顾名思义，它表示输入输出的两组iSlice在物理内存上是同一块，可以用于在iGraph中表达Reshape、Split、Concat等逻辑。`identity` 指令要求输入输出的 `iSlice` 表示同一块的物理内存。

```c
identity [ts], [ss]

// example
identity t1, s1 // Shape(t1) = (4, 64), Shape(s1) = (2, 128)
identity t1, [s1, s2] // Shape(t1) = (4, 64), Shape(s1) = Shape(s2) = (4, 32)
identity [t1, t2], s1 // Shape(t1) = Shape(t2) = (4, 32), Shape(s1) = (4, 64)
```