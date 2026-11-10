---
title: "Thread Mapping and Memory Layout"
date: 2026-09-21
excerpt: "Illustrating CUDA memory access through mathematical models of thread mapping and memory layout."
categories:
  - CUDA
tags:
  - CUDA
  - GEMM
  - Matrix Multiplication
toc: true
---

## Introduction

Understanding how threads are assigned to logical data elements, and how those elements are arranged in memory, is essential for reasoning about memory access patterns and designing efficient CUDA kernels.

All threads in a CUDA grid execute the same kernel function, using their block and thread indices to determine which logical elements to process. Depending on the algorithm, each thread may process one or multiple elements. This mapping is explicitly defined in the kernel.

## Memory Layout

In computing, row-major order and column-major order are two common methods for storing multidimensional arrays in linear storage. For a $d$-dimensional array with shape $N=(n_1, n_2, \dots, n_d)$, a given element of this array is specified by a tuple of zero-based indices $i = (i_1, i_2, \dots, i_d), \; i_k \in[0, n_k - 1]$.
In row-major order, the last dimension is contiguous, so that the memory offset of this element is given by:
$$
O_r(i) = \sum_{k = 1}^di_k(\prod_{l=k+1}^d n_l) = i_d+n_d \cdot(i_{d-1} + n_{d-1}\cdot(i_{d-2} + n_{d-2}\cdot(\cdots +n_{2}i_1)\cdots))
$$
In column-major order, the first dimension is contiguous, so that the memory offset of this element is given by:
$$
O_c(i) = \sum_{k = 1}^d i_k (\prod_{l=1}^{k-1} n_l) = i_1+n_1 \cdot(i_{2} + n_{2}\cdot(i_{3} + n_{3}\cdot(\cdots +n_{d-1}i_d)\cdots))
$$
where the empty product is defined to be 1.

Both layouts can be expressed using the same stride-based formula:
$$
O(i) = \sum_{k=1}^{d}i_ks_k
$$
where $s_k$ is the stride of dimension k. 
We can therefore describe a layout using a pair of tuples: the shape $N$, which specifies the range of valid coordinates, and the stride $S$, which determines how those coordinates map to linear offsets.
$$
L = N:S, \; N = (n_1, n_2, \dots, n_d), \; S = (s_1, s_2, \dots, s_d)
$$
Consider a multidimensional array $A$ of shape $(2, 3, 4)$. In row-major order, the stride is $S_r = (3 \times 4, 4, 1)=(12, 4, 1)$. In column-major order, it is $S_c = (1, 2, 2\times3)=(1,2,6)$.
![](/images/Pasted%20image%2020261021174437.png)
## Thread Mapping

Thread mapping and memory layout describe two distinct aspects of data access. Thread mapping determines which logical elements are assigned to each thread, while memory layout determines where those elements are stored. Neither uniquely determines the other.

Let $\mathcal{T}$ denote the thread index space and $\mathcal{I}$ denote the logical coordinate space of a $d$-dimensional tensor with shape $N=(n_1,n_2,\ldots,n_d)$. For a CUDA block containing $n_{\text{threads}}$ threads, the thread index space can be represented as: 
$$\mathcal{T}=\{0,1,\ldots,n_{\text{threads}}-1\}$$
The corresponding logical coordinate space is defined as:
$$
\mathcal{I}=\prod_{k=1}^{d}{\{0,1,\ldots,n_k-1\}}
$$
A thread may be responsible for one or multiple logical elements. To describe this more generally, we introduce a value index space $\mathcal{V}$, which indexes the elements assigned to each thread.

The thread mapping can then be defined as:
$$
\Phi:\mathcal{T}\times\mathcal{V}\rightarrow\mathcal{I}
$$
For a thread index $t\in\mathcal{T}$ and a value index $v\in\mathcal{V}$, the mapping produces a logical coordinate:
$$
\Phi(t,v)=\bigl(i_1(t,v),i_2(t,v),\ldots,i_d(t,v)\bigr).

$$
Thus, the mapping describes which logical element is associated with each thread-value pair.
Let $i_k(t,v)$ denote the $k$-th coordinate of $\Phi(t,v)$. Composing the thread mapping with the memory layout $L=N:S$ gives the linear memory offset associated with each thread-value pair:
$$
\begin{aligned}
O_{\mathrm{thread}}(t,v)
&=(L\circ\Phi)(t,v)
=L(\Phi(t,v))
\end{aligned}
$$
## Practical Example

In tiled GEMM, each CUDA thread block computes a $BM \times BN$ output tile. At each step along the reduction dimension, the block cooperatively loads a $BM \times BK$ tile of $A$ and a $BK \times BN$ tile of $B$ from global memory into shared memory, then accumulates their product into the output tile.

![](/images/Pasted%20image%2020261108203031.png)

Let $M=N=K=1024$, $BM=BN=128$, and $BK=8$. All three matrices are stored in row-major order using `float`, and each block is one-dimension with 256 threads. Fix one block and one reduction step, and consider the copy of its $A$ tile from global memory to shared memory.

We assign each thread a group of $W=4$ consecutive elements along the $K$ dimension, allowing a `float4` load. Each tile row therefore contains $P=BK/W=2$ groups. Let $t \in \{0,\ldots,255\}$ be the linear thread index within the block and $v \in \{0,\ldots,W-1\}$ be the element index within the group. Thread $t$ selects row $\lfloor t/P\rfloor$ and group $t\bmod P$ within that row. Multiplying the group index by $W$ gives the starting column, and adding $v$ gives the element’s column coordinate. The mapping to tile-local coordinates $(m,k)\in\mathcal I=\{0,\ldots,127\}\times\{0,\ldots,7\}$ is therefore
$$
\begin{aligned}
\Phi(t,v)
&=\left(\left\lfloor\frac{t}{P}\right\rfloor,\;W(t\bmod P)+v\right)\\
&=\left(\left\lfloor\frac{t}{2}\right\rfloor,\;4(t\bmod2)+v\right)
\end{aligned}
$$

This assignment covers all $128\times8$ elements exactly once.

Both layouts take the same tile-local coordinate $(m,k)$ and return an offset in units of `float` elements from their respective base addresses.

In global memory, the tile is a view into $A$, so adjacent tile rows start $K=1024$ elements apart. If $(m_0,k_0)$ is the tile origin in $A$, the tile base is `A + m0 * K + k0` for a flat pointer `A`. The layout and offset relative to that base are
$$
\begin{aligned}
L_{\mathrm{global}} &= (BM,BK):(K,1)=(128,8):(1024,1)
\end{aligned}
$$
In shared memory, the logical element $(m,k)$ is stored at `As[k][m]` in the array `float As[BK][BM]`. Consecutive values of $m$ occupy consecutive array elements. Relative to `&As[0][0]`, the layout and offset are

$$
\begin{aligned}
L_{\mathrm{shared}} &= (BM,BK):(1,BM)=(128,8):(1,128)
\end{aligned}
$$
Composing each layout with the same thread mapping gives the source and destination offsets for each $(t,v)$.
$$
\begin{aligned}
O_{\mathrm{thread,global}}(t,v)
&=(L_{\mathrm{global}}\circ\Phi)(t,v) \\
&= K\left\lfloor\frac{t}{2}\right\rfloor+4(t\bmod 2)+v\\
&=1024\left\lfloor\frac{t}{2}\right\rfloor+4(t\bmod 2)+v\\
O_{\mathrm{thread,shared}}(t,v)
&=(L_{\mathrm{shared}}\circ\Phi)(t,v) \\
&=\left\lfloor\frac{t}{2}\right\rfloor+BM\bigl(4(t\bmod 2)+v\bigr)\\
&=\left\lfloor\frac{t}{2}\right\rfloor+128\bigl(4(t\bmod 2)+v\bigr)
\end{aligned}
$$

The formulas above translate directly into the following CUDA code. The thread mapping determines each thread’s tile-local coordinates, and the two layouts determine the corresponding source and destination offsets.

```cpp
constexpr int K = 1024;
constexpr int BM = 128;
constexpr int BK = 8;
constexpr int W = 4;
__shared__ float As[BK * BM];

const int aRowL = threadIdx.x / (BK / W);
const int aRowG = blockIdx.y * BM + aRowL;
const int aCol = threadIdx.x % (BK / W) * W;

const float4 group = *reinterpret_cast<const float4*>(&A[aRowG * K + r * BK + aCol]); // r: reduction step
As[(aCol + 0) * BM + aRowL] = group.x;
As[(aCol + 1) * BM + aRowL] = group.y;
As[(aCol + 2) * BM + aRowL] = group.z;
As[(aCol + 3) * BM + aRowL] = group.w;

```