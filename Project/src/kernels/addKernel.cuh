#include "../common.cuh"

// This kernel takes two input vectors (op1* and op2*) then adds them together and stored the result in sum*
__global__ void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    //calculate the position of the thread in the vector
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    //verify that the thread is in bounds
    if (pos >= count) return;

    //make the addition
    sum[pos] = op1[pos] + op2[pos];
}

// This function is used to call the AddKernel
static void Add(const float *op1, const float *op2, int count, float *sum)
{
    // initialize gridDim and blockDim

    dim3 blockDim(256);
    dim3 gridDim((count + blockDim.x-1)/blockDim.x );
    //call kernel
    AddKernel<<<gridDim, blockDim>>>(op1, op2, count, sum);
}
