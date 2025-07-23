
# Chapter 2 — Exercises (Heterogeneous Data Parallel Computing)

## Question 1

If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (`i`)?

- (A) `i = threadIdx.x + threadIdx.y`
- (B) `i = blockIdx.x + threadIdx.x`
- (C) `i = blockIdx.x * blockDim.x + threadIdx.x`
- (D) `i = blockIdx.x * threadIdx.x`

    Ans **C**
---

## Question 2

Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (`i`) of the first element to be processed by a thread?

- (A) `i = blockIdx.x * blockDim.x + threadIdx.x + 2`
- (B) `i = blockIdx.x * blockDim.x + threadIdx.x * 2`
- (C) `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`
- (D) `i = blockIdx.x * blockDim.x * 2 + threadIdx.x`

    Ans **C** as each tread calculates 2 elements 
---

## Question 3

We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2 * blockDim.x` consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element.  
What variable should be the index for the first element to be processed by each thread? What would be the expression for calculating the global index of the first element?

- (A) `i = threadIdx.x + blockIdx.x * blockDim.x * 2`
- (B) `i = threadIdx.x + blockIdx.x * blockDim.x + 2`
- (C) `i = threadIdx.x + blockIdx.x * 2 + blockDim.x * 2`
- (D) `i = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.x * 2`

    Ans **A** as there are already `blockIdx.x * blockDim.x * 2` elements prior to the thread to process.

---

## Question 4

For a vector addition, assume that the vector length is 88000. Each thread calculates one output element. The maximum number of threads per block is 1024.  
Calculate the configuration of the kernel call to have a minimum number of thread blocks to cover all output elements.  
**How many threads will be in the grid?**

    threads = 1024
    blocks = ceil(88000 / 1024) = 86
    total allocated block = 86 * 1024

---

## Question 5

If you want to allocate an array of `N` integer elements in the CUDA device memory using `cudaMalloc()`, what would be an appropriate expression for the second argument of the `cudaMalloc()` call?

- (A) `N`
- (B) `sizeof(int)`
- (C) `N * sizeof(int)`
- (D) `&N`

    Ans **C**

---

## Question 6

If you want to allocate an array of `N` floating-point elements and have the pointer `f` point to the beginning of the allocated memory, what would be the correct expression?

- (A) `f = sizeof(float) * N`
- (B) `sizeof(float)`
- (C) `f = N`
- (D) `f = (float*) malloc(N * sizeof(float))`

    Ans **D**


---

## Question 7

If we want to copy 3000 bytes of data from host array `A_h` (`A_h` is a pointer to a source array) to device array `A_d` (`A_d` is a pointer to a destination array), what would be an appropriate API call for that data copy in CUDA?

- (A) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
- (B) `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyHostToDevice);`
- (C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyDeviceToHost);`
- (D) `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`


    Ans **A** `cudaMemcpy(destination, source, size_in_bytes, direction);`

---

## Question 8

How would one declare a variable `err` that can appropriately receive the return value of a CUDA API call?

- (A) `int err;`
- (B) `cudaError err;`
- (C) `cudaSuccess err;`
- (D) `cudaError_t err;`

    Ans **D**
---

## Question 9

Consider the following CUDA kernel and the corresponding host function that calls it:

```cpp
__global__ void foo_kernel(float *a, float *b, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = 2.7f * a[i] - 4.3f;
}

void foo(float *a, float *b) {
  unsigned int N = 28000;
  foo_kernel <<< 28000/128, 128 >>> (a, b, N);
}
```
### Answer the following questions:

1) What is the number of threads per block? **128**
1) What is the number of blocks in the grid? **219** ceil(28000/128)
1) What is the number of threads in the grid? **28032** 219 * 128
1) What is the number of threads that execute the code on line 02? **28032** for total no of time the loops run i.e. for each element of the grid
1) What is the number of threads that execute the code in line 04? **28000** filter by the boundary cond

---

## Question 10
A team member is very frustrated with CUDA. He has been complaining that CUDA doesn't let him write host and device functions that he plans to execute on both the host and the device. He now writes one function as a host function and once as a device function.
What is your response?

    Ans we can declare an function as both __host__ and __device__ to achive the objective. Like the following `__host__ __device__ void foo_kernel(float *a, float *b, unsigned int n)`