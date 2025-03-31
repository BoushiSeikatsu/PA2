#ifndef KERNELS_H
#define KERNELS_H

#include "cnn.h"
#include <cuda_runtime.h>

// --- Forward Pass Kernels ---

__global__ void relu_forward_kernel(float* output, const float* input, int N);

__global__ void fc_forward_kernel(float* output,
                                  const float* input,
                                  const float* weights,
                                  const float* biases,
                                  int batch_size,
                                  int in_features,
                                  int out_features);

// Placeholder - complex implementation omitted
__global__ void conv_forward_kernel(float* output,
                                    const float* input,
                                    const float* weights,
                                    const float* biases,
                                    int batch_size,
                                    int in_h, int in_w, int in_c,
                                    int out_h, int out_w, int out_c, // out_c = num_filters
                                    int kernel_size, int stride, int padding);

// Placeholder - complex implementation omitted
// Needs d_pool_indices buffer to store max locations
__global__ void pool_forward_kernel(float* output,
                                    int* max_indices, // Output: indices of max vals
                                    const float* input,
                                    int batch_size,
                                    int in_h, int in_w, int in_c,
                                    int out_h, int out_w, // out_c = in_c
                                    int pool_size, int stride);


__global__ void softmax_kernel(float* output, const float* input, int batch_size, int num_classes);


// --- Loss & Initial Gradient ---

__global__ void softmax_cross_entropy_loss_backward_kernel(float* loss, // Output loss (per sample or single value)
                                                           float* delta, // Output delta (gradient w.r.t softmax *input*)
                                                           const float* softmax_output, // Input from softmax_kernel
                                                           const int*   true_labels,
                                                           int batch_size,
                                                           int num_classes);

// --- Backward Pass Kernels ---

__global__ void relu_backward_kernel(float* delta_in, // Output: Gradient w.r.t ReLU input
                                     const float* delta_out, // Input: Gradient w.r.t ReLU output
                                     const float* activation_in, // Input: Activations *before* ReLU
                                     int N);

// Calculates gradients w.r.t FC input, weights, and biases
__global__ void fc_backward_kernel(float* delta_in,         // Output: Gradient w.r.t layer input
                                   float* grad_weights,     // Output: Gradient w.r.t weights
                                   float* grad_biases,      // Output: Gradient w.r.t biases
                                   const float* delta_out,  // Input: Gradient w.r.t layer output
                                   const float* activation_in, // Input: Activations fed *into* this layer
                                   const float* weights,    // Input: Layer weights
                                   int batch_size,
                                   int in_features,
                                   int out_features);

// Placeholder - very complex implementation omitted
__global__ void conv_backward_kernel(float* delta_in,
                                     float* grad_weights,
                                     float* grad_biases,
                                     const float* delta_out,
                                     const float* activation_in,
                                     const float* weights,
                                     int batch_size,
                                     int in_h, int in_w, int in_c,
                                     int out_h, int out_w, int out_c,
                                     int kernel_size, int stride, int padding);


// Placeholder - complex implementation omitted
// Needs d_pool_indices read from forward pass
__global__ void pool_backward_kernel(float* delta_in,
                                     const float* delta_out,
                                     const int* max_indices, // Input: Indices from forward pass
                                     int batch_size,
                                     int in_h, int in_w, int in_c,
                                     int out_h, int out_w, // out_c = in_c
                                     int pool_size, int stride);


// --- Weight Update Kernel ---

__global__ void sgd_update_kernel(float* weights,
                                  float* biases,
                                  const float* grad_weights,
                                  const float* grad_biases,
                                  float learning_rate,
                                  size_t num_weights,
                                  size_t num_biases);

// --- Helper Device Functions (if needed) ---
// __device__ float activation_function(...)

#endif // KERNELS_H