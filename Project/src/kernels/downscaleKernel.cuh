#include "../common.cuh"


// width:   output image width
// height:  output image height
// stride:  stride size of input image
// out:     resulting array (image)
// src:     input array (image)

__global__ void DownscaleKernel(int width, int height, int stride, float *out, const float* src)
{
    // variables for the x and y coordinates of the pixel that the thread is mapped to
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // Exit if the thread is out of bounds of the image
    if (ix >= width-1 || iy >= height-1)
        return;

    // These lines calculate the corresponding coordinates in the original image (src) for the downsampled image (out). 
    // Each pixel in the downsampled image corresponds to a 2x2 block of pixels in the original image.
    const size_t srcx = ix * 2;
    const size_t srcy = iy * 2;

    // This line computes the downsampled value at the current position (ix, iy) in the output array out. 
    // It takes the average of the 2x2 block of pixels in the original image centered at (srcx, srcy).
    out[ix + iy * stride] = 0.25f * (src[srcx + srcy*stride] + src[srcx + (srcy+1)*stride] +
                                    src[(srcx+1) + srcy*stride] + src[(srcx+1) + (srcy+1)*stride]);
}

// width:   input image width
// height:  input image height
// stride:  stride size of input image
// out:     resulting array (image)
// src:     input array (image)
// newWidth:  output image width
// newHeight: output image height
static
void Downscale(const float *src, int width, int height, int stride,
               int newWidth, int newHeight, int newStride, float *out)
{

    dim3 blockDim(32, 8);

    // Calculate the number of blocks needed in each dimension
    int gridDimX = (newWidth + blockDim.x - 1) / blockDim.x;
    int gridDimY = (newHeight + blockDim.y - 1) / blockDim.y;

    dim3 gridDim(gridDimX, gridDimY);

    DownscaleKernel<<<gridDim, blockDim>>>(newWidth, newHeight, newStride, out, src);
}
