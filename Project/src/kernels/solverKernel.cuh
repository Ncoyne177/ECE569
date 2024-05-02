#include "../common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/// It is one iteration of Jacobi method for a corresponding linear system.
/// Template parameters are describe CTA size
// du0:     current horizontal displacement approximation
// dv0:     current vertical displacement approximation
// Ix:      image x derivative
// Iy:      image y derivative
// Iz:      temporal derivative
// w:       width
// h:       height
// s:       stride
// alpha:   degree of smoothness
// du1:     new horizontal displacement approximation
// dv1:     new vertical displacement approximation

__global__
void JacobiIteration(const float *du0,
                     const float *dv0,
                     const float *Ix,
                     const float *Iy,
                     const float *Iz,
                     int w, int h, int s,
                     float alpha,
                     float *du1,
                     float *dv1)
{
    // create shared memory
    extern __shared__ float shared_mem[];
    float* du = shared_mem;
    float* dv = (float*)&du[(blockDim.x+2) * (blockDim.y+2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (blockDim.x + 2);

    // Load data to shared memory.
    // load tile being processed
    du[shMemPos] = du0[pos];
    dv[shMemPos] = dv0[pos];

    // load necessary neighbouring elements
    // We clamp out-of-range coordinates.
    // It is equivalent to mirroring
    // because we access data only one step away from borders.
    if (threadIdx.y == 0)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        x = min(bsx + threadIdx.x, w - 1);
        // row just below the tile
        y = max(bsy - 1, 0);
        gmPos = y * s + x;
        smPos = threadIdx.x + 1;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];

        // row above the tile
        y = min(bsy + blockDim.y, h - 1);
        smPos += (blockDim.y + 1) * (blockDim.x + 2);
        gmPos  = y * s + x;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
    }
    else if (threadIdx.y == 1)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        y = min(bsy + threadIdx.x, h - 1);
        // column to the left
        x = max(bsx - 1, 0);
        smPos = blockDim.x + 2 + threadIdx.x * (blockDim.x + 2);
        gmPos = x + y * s;

        // check if we are within tile
        if (threadIdx.x < blockDim.y)
        {
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            // column to the right
            x = min(bsx + blockDim.x, w - 1);
            gmPos  = y * s + x;
            smPos += blockDim.x + 1;
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
        }
    }

    __syncthreads();

    if (ix >= w || iy >= h) return;

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + blockDim.x + 2;
    down  = shMemPos - blockDim.x - 2;

    float sumU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
    float sumV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;

    float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos])
                 / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = sumU - Ix[pos] * frac;
    dv1[pos] = sumV - Iy[pos] * frac;
}



/// It is one iteration of Jacobi method for a corresponding linear system.
// du0     current horizontal displacement approximation
// dv0     current vertical displacement approximation
// Ix      image x derivative
// Iy      image y derivative
// Iz      temporal derivative
// w       width
// h       height
// s       stride
// alpha   degree of smoothness
// du1     new horizontal displacement approximation
// dv1     new vertical displacement approximation

static
void SolveForUpdate(const float *du0,
                    const float *dv0,
                    const float *Ix,
                    const float *Iy,
                    const float *Iz,
                    int w, int h, int s,
                    float alpha,
                    float *du1,
                    float *dv1)
{
    // block size
    dim3 blockDim(32, 6);
    dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);

    // determine shared memory size and call the kernel
    size_t shared_mem_size = 2 * (blockDim.x+2) * (blockDim.y+2) * sizeof(float);

    JacobiIteration<<<gridDim, blockDim, shared_mem_size>>>(du0, dv0, Ix, Iy, Iz,
                                               w, h, s, alpha, du1, dv1);
}
