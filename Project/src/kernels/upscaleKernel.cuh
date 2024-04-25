#include "../common.cuh"

// upscale one component of a displacement field, CUDA kernel
// width   width of out
// height  height of out
// stride  stride size of out
// scale   scale factor (multiplier)
// out     resulting array
// src     input array

__global__ void UpscaleKernel(int width, int height, int stride, float scale, float *out, const float* src)
{   
    // calculate the x and y position of the thread
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    //verify that the thread is within the bounds of the array
    if (ix >= width || iy >= height) return;

    // calculate the indexing for the input vector
    float x = ((float)ix - 0.5f) * 0.5f;
    float y = ((float)iy - 0.5f) * 0.5f;
    const size_t index = x + y*stride;


    // scale input vector and store the value into the output vector
    out[ix + iy * stride] = src[index] * scale;
}

// upscale one component of a displacement field, kernel wrapper
// src         field component to upscale
// width       width of src
// height      height of src
// stride      stride size of src
// newWidth    width of out
// newHeight   height of out
// newStride   stride size of out
// scale       value scale factor (multiplier)
// out         upscaled field component

static void Upscale(const float *src, int width, int height, int stride,
             int newWidth, int newHeight, int newStride, float scale, float *out)
{
    dim3 blockDim(32, 8);
    dim3 gridDim((newWidth + blockDim.x - 1) / blockDim.x, (newHeight + blockDim.y - 1) / blockDim.y);
    UpscaleKernel<<<gridDim, blockDim>>>(newWidth, newHeight, newStride, scale, out, src);
}
