#include <cuda_runtime.h>
#include <iostream>

constexpr unsigned int THREADS_PER_BLOCK_DIM = 8; // 8x8 threads in a block

cudaError_t error = cudaSuccess;

__global__ void fillData(const unsigned int pitch, const unsigned int rows, const unsigned int cols, float* data)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds to avoid out-of-bounds memory access
    if (x < cols && y < rows)
    {
        // Calculate the offset to the correct element based on pitch
        float* row = (float*)((char*)data + y * pitch);
        row[x] = y * cols + x; // Fill data with incremental values
    }
}

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaSetDevice(0);

    float* devPtr;
    float* hostPtr;
    size_t pitch;

    const unsigned int mRows = 5;
    const unsigned int mCols = 10;

    // Allocate pitch memory with alignment using cudaMallocPitch
    cudaError_t err = cudaMallocPitch(&devPtr, &pitch, mCols * sizeof(float), mRows);
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed!" << std::endl;
        return -1;
    }

    // Prepare grid and blocks (2D grid of 2D blocks of size 8x8)
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIM, THREADS_PER_BLOCK_DIM);
    dim3 numBlocks((mCols + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM,
        (mRows + THREADS_PER_BLOCK_DIM - 1) / THREADS_PER_BLOCK_DIM);

    // Launch the kernel
    fillData << <numBlocks, threadsPerBlock >> > (pitch, mRows, mCols, devPtr);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Allocate host memory for the matrix
    hostPtr = (float*)malloc(mRows * mCols * sizeof(float));

    // Copy data back from device to host using cudaMemcpy2D
    cudaMemcpy2D(hostPtr, mCols * sizeof(float), devPtr, pitch, mCols * sizeof(float), mRows, cudaMemcpyDeviceToHost);

    // Check the data by printing out the matrix
    std::cout << "Matrix (after kernel incrementing):" << std::endl;
    for (unsigned int i = 0; i < mRows; ++i)
    {
        for (unsigned int j = 0; j < mCols; ++j)
        {
            std::cout << hostPtr[i * mCols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device and host memory
    cudaFree(devPtr);
    free(hostPtr);

    return 0;
}
