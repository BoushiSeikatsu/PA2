#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- MNIST Loading Placeholders ---
// In a real project, implement these to load data from files
bool load_mnist_images(const char* filename, std::vector<float>& images, int& num_images, int& height, int& width);
bool load_mnist_labels(const char* filename, std::vector<int>& labels, int& num_labels);

// --- Weight Initialization ---
void initialize_weights_xavier(float* weights, int in_features, int out_features);
void initialize_biases(float* biases, int num_features);


#endif // UTILS_H