#include "../common.cuh"


// warp image with a given displacement field
// width   image width
// height  image height
// stride  image stride
// u       horizontal displacement
// v       vertical displacement
// out     result image
// src     input image
__global__ void WarpingKernel(int width, int height, int stride,
                              const float *u, const float *v, float *out, const float* src)
{
    // calculate the x and y index of the pixel
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // find the position of the pixel in the 1D array representation
    const int pos = ix + iy * stride;

    // ignore threads that are out of bounds
    if (ix >= width || iy >= height) return;

    // 
    float x = ((float)ix + u[pos] + 0.5f) / (float)width;
    float y = ((float)iy + v[pos] + 0.5f) / (float)height;
    const int index = x + y * stride;

    out[pos] = src[index];
}


/// For each output pixel there is a vector which tells which pixel
/// from a source image should be mapped to this particular output
/// pixel.
/// It is assumed that images and the vector field have the same stride and
/// resolution.
// src source image
// w   width
// h   height
// s   stride
// u   horizontal displacement
// v   vertical displacement
// out warped image
// src input image

static
void WarpImage(const float *src, int w, int h, int s,
               const float *u, const float *v, float *out)
{
    dim3 blockDim(32, 6);
    int gridDimX = (w + blockDim.x - 1) / blockDim.x;
    int gridDimY = (h + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridDimX, gridDimY);


    WarpingKernel<<<gridDim, blockDim>>>(w, h, s, u, v, out, src);
}
