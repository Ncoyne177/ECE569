#include "../common.cuh"
//this file is for computing image spatial derivatives
// Spacial derivatives define the rate of change of intensity in the image with respect to spatial coordinates (x and y)


// width   image width
// height  image height
// stride  number of elements in a row
// Ix      x derivative
// Iy      y derivative
// Iz      temporal derivative

__global__ void ComputeDerivativesKernel(int width, int height, int stride,
                                         float *Ix, float *Iy, float *Iz,
                                         const float* src, const float* target)
{   
    //idy is the y index and idx is the x index
    int idy = blockIdx.y * blockDim.y+threadIdx.y;
    int idx = blockIdx.x * blockDim.x+threadIdx.x;


    extern __shared__ float shared_mem[];

    //position of the current pixel
    const int pos = idx + idy*stride;
    const int tid = threadIdx.x + threadIdx.y*blockDim.x;
    // Allocate shared memory for src and target
    float* shared_src = shared_mem;
    float* shared_target = (float*)&shared_src[blockDim.x*blockDim.y];

    // Copy src and target into shared memory
    if (idx < width && idy < height)
    {
        shared_src[tid] = src[pos];
        shared_target[tid] = target[pos];
    }
    __syncthreads();
    
    //Threads at the image borders (within a 2-pixel boundary) are excluded from computation to avoid accessing out-of-bounds memory.
    if (idx >= width-2 || idy >= height-2) return;
    // For each pixel, we use a central difference formula to estimate the derivative. 
    // We calculate t0 and t1 separately for the source (src) and target (target) images.
    // t0 and t1 are weighted sums of neighboring pixels.The weights for the central differences are 8.0 and -1.0, and the sum is divided by 12.0 to normalize the result.
    // Finally, we take the average of t0 and t1 to compute the x-derivative (Ix) for the current pixel.

    float t0, t1;
    //if statements prevent the thread from accessing shared memory that is out of bounds
    if (threadIdx.x < 2){
        t0  = src[(idx - 2) + idy*stride ];
        t0 -= src[(idx - 1) + idy*stride ] * 8.0f;

        t1  = target[(idx - 2) + idy*stride ];
        t1 -= target[(idx - 1) + idy*stride ] * 8.0f;
    }
    else{
        t0  = shared_src[tid-2 ];
        t0 -= shared_src[tid-1 ] * 8.0f;

        t1  = shared_target[tid-2 ];
        t1 -= shared_target[tid-1 ] * 8.0f;
    }
    if (threadIdx.x >(blockDim.x-2)){
        t0 += src[(idx + 1) + idy*stride ] * 8.0f; 
        t0 -= src[(idx + 2) + idy*stride ];

        t1 += target[(idx + 1) + idy*stride ] * 8.0f; 
        t1 -= target[(idx + 2) + idy*stride ];
    }
    else{
        t0 += shared_src[tid+1] * 8.0f; 
        t0 -= shared_src[tid+2];

        t1 += shared_target[tid+1] * 8.0f; 
        t1 -= shared_target[tid+2];
    }

    t0 /= 12.0f;
    t1 /= 12.0f;

    Ix[pos] = (t0 + t1) * 0.5f;

    // temporal derivative simply finds the difference in time
    Iz[pos] = shared_target[tid ] - shared_src[tid ];

    // y derivative
    if (threadIdx.y < 2){
        t0  = src[idx + (idy - 2)*stride ];
        t0 -= src[idx + (idy - 1)*stride ] * 8.0f;

        t1  = target[idx + (idy - 2)*stride ];
        t1 -= target[idx + (idy - 1)*stride ] * 8.0f;
    }
    else{
        t0  = shared_src[tid - 2*blockDim.x ];
        t0 -= shared_src[tid - 1*blockDim.x ] * 8.0f;

        t1  = shared_target[tid - 2*blockDim.x ];
        t1 -= shared_target[tid - 1*blockDim.x ] * 8.0f;
    }
    if (threadIdx.y >(blockDim.y-2)){
        t0 += src[idx + (idy + 1)*stride ] * 8.0f; 
        t0 -= src[idx + (idy + 2)*stride ];

        t1 += target[idx + (idy + 1)*stride ] * 8.0f; 
        t1 -= target[idx + (idy + 2)*stride ];
    }
    else{
        t0 += shared_src[tid + 1*blockDim.x ] * 8.0f; 
        t0 -= shared_src[tid + 2*blockDim.x ];

        t1 += shared_target[tid + 1*blockDim.x] * 8.0f; 
        t1 -= shared_target[tid + 2*blockDim.x ];
    }


    t0 /= 12.0f;
    t1 /= 12.0f;

    Iy[pos] = (t0 + t1) * 0.5f;
}




//I0  source image
//I1  tracked image
//w   image width
//h   image height
//s   number of elements in a idy (stride)
//Ix  x derivative
//Iy  y derivative
//Iz  temporal derivative
static
void ComputeDerivatives(const float *I0, const float *I1,
                        int w, int h, int s,
                        float *Ix, float *Iy, float *Iz)
{
    dim3 blockDim(32, 6);

    int gridDimX = (w + blockDim.x - 1) / blockDim.x;
    int gridDimY = (h + blockDim.y - 1) / blockDim.y;

    dim3 gridDim(gridDimX, gridDimY);

    // determine shared memory size and call the kernel
    size_t shared_mem_size = 2 * blockDim.x * blockDim.y * sizeof(float);
    ComputeDerivativesKernel<<<gridDim, blockDim, shared_mem_size>>>(w, h, s, Ix, Iy, Iz, I0, I1);
}
