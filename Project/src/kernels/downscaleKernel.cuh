#include "../common.cuh"


// width:   input image width
// height:  input image height
// stride:  stride size of input image
// out:     resulting array (image)
// src:     input array (image)

__global__ void DownscaleKernel(int width, int height, int stride, int newStride, float *out, const float* src)
{
    // variables for the x and y coordinates of the pixel that the thread is mapped to
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Exit if the thread is out of bounds of the image
    if (ix >= width-1 || iy >= height-1)
        return;

    // These lines calculate the corresponding coordinates in the down sampled image (out)
    // Each pixel in the downsampled image corresponds to a 2x2 block of pixels in the original image.
    float outx = ((float)ix - 0.5f)* 0.5f;
    float outy = ((float)iy - 0.5f)* 0.5f;
    const size_t index = outx + outy*stride;
    // the tmp variable stores a downscaled version of the target pixel in the input image
    const float tmp = 0.25f * src[ix + iy*stride];
    // The output image will be updated by four different threads, so an atomicAdd function is used
    atomicAdd(&(out[index]), tmp);
}

// width:   input image width
// height:  input image height
// stride:  stride size of input image
// out:     resulting array (image)
// src:     input array (image)
// newWidth:  output image width
// newHeight: output image height
// newStride: stride of the output image
static
void Downscale(const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out)
{

    dim3 blockDim(32, 8);

    // Calculate the number of blocks needed in each dimension
    int gridDimX = (width + blockDim.x - 1) / blockDim.x;
    int gridDimY = (height + blockDim.y - 1) / blockDim.y;

    dim3 gridDim(gridDimX, gridDimY);

    DownscaleKernel<<<gridDim, blockDim>>>(width, height, stride, newStride, out, src);
}
