#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Define the spatial filter function
__device__ float get_spatial_filter(int sf, float x, float y, float sigma) {
    // Implementing the provided formula for the spatial filter

    // Calculate Hermite polynomials
    float Hn_x = 1.0f;
    float Hn_y = 1.0f;
    for (int i = 1; i <= sf; i++) {
        Hn_x *= x / (sqrt(2 * sigma));
        Hn_y *= y / ((sigma) * sqrt(2));
    }

    // Calculate the exponent term
    float exponent = pow(-1 / ((sigma) * sqrt(2)), 2 * sf);

    // Calculate the Gaussian component
    float gaussian = exp(-(x * x + y * y) / (2 * sigma * sigma)) / ((sigma) * sqrt(2 * M_PI));

    // Combine all terms
    return Hn_x * Hn_y * exponent * gaussian;
}

// 2D convolution function
__device__ float conv2D(float *input, int width, int height, int x, int y, float *filter, int filterSize) {
    float result = 0.0f;
    int filterRadius = filterSize / 2;

    for (int i = -filterRadius; i <= filterRadius; i++) {
        for (int j = -filterRadius; j <= filterRadius; j++) {
            int inputX = x + i;
            int inputY = y + j;

            // Clamp input indices to image bounds
            inputX = min(max(inputX, 0), width - 1);
            inputY = min(max(inputY, 0), height - 1);

            int filterIndex = (i + filterRadius) * filterSize + (j + filterRadius);
            int inputIndex = inputY * width + inputX;

            result += input[inputIndex] * filter[filterIndex];
        }
    }

    return result;
}

// CUDA kernel function
__global__ void spat_filt_kernel(float *temp_filt, int nFrames, int L, int nTemp_filters, int nSpat_filters, float *spat_filt, float sigma) {
    int tf = blockIdx.x * blockDim.x + threadIdx.x; // Temporal filter index
    int sf = blockIdx.y * blockDim.y + threadIdx.y; // Spatial filter index

    if (tf < nTemp_filters && sf < nSpat_filters) {
        // Get spatial filter for this thread
        float x = blockIdx.z * blockDim.z + threadIdx.z; // x-coordinate within the filter window
        float y = blockIdx.z * blockDim.z + threadIdx.z; // y-coordinate within the filter window
        float spatial_filter = get_spatial_filter(sf, x, y, sigma); 

        for (int fr = 0; fr <= nFrames - L; fr++) {
            // Get current frame
            float *frame = &temp_filt[(tf * (nFrames - L + 1) + fr)];

            // Iterate over pixels within frame
            for (int p = 0; p < frame_size; p++) {
                // Convolve pixel with spatial filter
                spat_filt[sf * nTemp_filters * frame_size + tf * frame_size + p] = conv2D(frame, width, height, x, y, spatial_filter, filterSize);
            }
        }
    }
}

int main() {
    // Define dimensions
    int nFrames = 100; // Example value
    int L = 5; // Example value
    int nTemp_filters = 3; // Example value
    int nSpat_filters = 2; // Example value
    int frame_size = 10; // Example value
    float sigma = 1.0f; // Example value
    int width = 1280; // Example value
    int height = 720; // Example value
    int filterSize = 3; // Example value

    // Allocate memory for temp_filt and spat_filt
    float *temp_filt, *spat_filt;
    cudaMallocManaged(&temp_filt, nTemp_filters * (nFrames - L + 1) * frame_size * sizeof(float));
    cudaMallocManaged(&spat_filt, nSpat_filters * nTemp_filters * frame_size * sizeof(float));

    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((nTemp_filters + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (nSpat_filters + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                       (frame_size + threadsPerBlock.z - 1) / threadsPerBlock.z);
    spat_filt_kernel<<<blocksPerGrid, threadsPerBlock>>>(temp_filt, nFrames, L, nTemp_filters, nSpat_filters, spat_filt, sigma);
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Output spat_filt or do further processing
    // Note: spat_filt contains the result of convolutions

    // Free allocated memory
    cudaFree(temp_filt);
    cudaFree(spat_filt);

    return 0;
}