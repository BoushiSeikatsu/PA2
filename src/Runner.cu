#include <cudaDefs.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Constants for discretization
#define THRESHOLD 100 // Example threshold for discretization
#define NUM_ELEMENTS 100 // Just an example for n (size of vector)

// CUDA kernel to discretize the matrix (converts to uint8_t)
__global__ void discretize_matrix(float* M0, uint8_t* M1, int m, int n, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        // Normalize and discretize values to uint8_t
        float value = M0[idx];
        float normalized_value = (value - min_val) / (max_val - min_val); // Normalize to [0, 1]
        M1[idx] = static_cast<uint8_t>(normalized_value * 255); // Discretize to uint8_t
    }
}

// CUDA kernel to compute Euclidean distance from the origin
__global__ void compute_distance(uint8_t* M1, float* distances, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float dist = 0.0f;
        for (int j = 0; j < n; j++) {
            dist += powf(M1[idx * n + j], 2); // Sum of squares (Euclidean distance)
        }
        distances[idx] = sqrtf(dist); // Square root to get the distance
    }
}

// CUDA kernel to find the farthest object (max distance)
__global__ void find_farthest(float* distances, int* farthest_idx, int m) {
    __shared__ float shared_distances[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        shared_distances[threadIdx.x] = distances[idx];
        __syncthreads();

        // Parallel reduction to find max distance
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (threadIdx.x % (2 * stride) == 0) {
                shared_distances[threadIdx.x] = max(shared_distances[threadIdx.x], shared_distances[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicMax(farthest_idx, shared_distances[0]);
        }
    }
}

// CUDA kernel to find the farthest element from the found object
__global__ void find_farthest_element(uint8_t* M1, float* distances, int farthest_idx, int n, int* farthest_element_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_dist = -1.0f;
        int element_idx = -1;
        for (int j = 0; j < n; j++) {
            float dist = static_cast<float>(M1[farthest_idx * n + j]);
            if (dist > max_dist) {
                max_dist = dist;
                element_idx = j;
            }
        }
        *farthest_element_idx = element_idx;
    }
}

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

int main() {
    initializeCUDA(deviceProp);
    const int m = 2 << 20;  // m > 2^20
    const int n = NUM_ELEMENTS; // Example n

    // Initialize the matrix with random values (real numbers)
    std::vector<float> M0(m * n);
    for (int i = 0; i < m * n; i++) {
        M0[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // Random float between 0 and 1
    }

    // Define min and max for normalization
    float min_val = *std::min_element(M0.begin(), M0.end());
    float max_val = *std::max_element(M0.begin(), M0.end());

    // Allocate memory for the discretized matrix and distances on the GPU
    uint8_t* M1;
    float* distances;
    int* farthest_idx;
    int* farthest_element_idx;

    cudaMalloc(&M1, m * n * sizeof(uint8_t));
    cudaMalloc(&distances, m * sizeof(float));
    cudaMalloc(&farthest_idx, sizeof(int));
    cudaMalloc(&farthest_element_idx, sizeof(int));

    // Copy matrix M0 to device memory
    cudaMemcpy(M1, M0.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Call the discretize kernel
    int block_size = 256;
    int grid_size = (m * n + block_size - 1) / block_size;
    discretize_matrix << <grid_size, block_size >> > (M0.data(), M1, m, n, min_val, max_val);
    cudaDeviceSynchronize();

    // Compute the distance from the origin
    grid_size = (m + block_size - 1) / block_size;
    compute_distance << <grid_size, block_size >> > (M1, distances, m, n);
    cudaDeviceSynchronize();

    // Find the farthest object

    find_farthest << <grid_size, block_size >> > (distances, farthest_idx, m);
    cudaDeviceSynchronize();

    // Find the farthest element from the farthest object
    int farthest_idx_host;
    cudaMemcpy(&farthest_idx_host, farthest_idx, sizeof(int), cudaMemcpyDeviceToHost);
    find_farthest_element << <1, 1 >> > (M1, distances, *farthest_idx, n, farthest_element_idx);
    cudaDeviceSynchronize();

    int farthest_element_idx_host;
    cudaMemcpy(&farthest_element_idx_host, farthest_element_idx, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Farthest element index: " << farthest_element_idx_host << std::endl;

    // Cleanup
    cudaFree(M1);
    cudaFree(distances);
    cudaFree(farthest_idx);
    cudaFree(farthest_element_idx);

    return 0;
}
