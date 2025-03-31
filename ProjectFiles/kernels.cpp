#include "kernels.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> // For expf
#include <algorithm> // For max

// Simple ReLU Forward
__global__ void relu_forward_kernel(float* output, const float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = max(0.0f, input[idx]);
    }
}

// Simple ReLU Backward
__global__ void relu_backward_kernel(float* delta_in,
                                     const float* delta_out,
                                     const float* activation_in,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        delta_in[idx] = (activation_in[idx] > 0.0f) ? delta_out[idx] : 0.0f;
    }
}

// Simplified FC Forward (No shared memory tiling for brevity)
// WARNING: Inefficient for large layers! Real implementation needs tiling.
__global__ void fc_forward_kernel(float* output,
                                  const float* input,
                                  const float* weights,
                                  const float* biases,
                                  int batch_size,
                                  int in_features,
                                  int out_features) {
    int batch_idx = blockIdx.y; // Map blocks to batch samples
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // Map threads to output features

    if (batch_idx < batch_size && out_idx < out_features) {
        float sum = 0.0f;
        // Input for this sample starts at input[batch_idx * in_features]
        const float* current_input = input + batch_idx * in_features;
        // Weights for this output feature start at weights[out_idx * in_features]
        // Note: Assumes weights are [out_features x in_features]
        const float* current_weights = weights + out_idx * in_features;

        for (int k = 0; k < in_features; ++k) {
            sum += current_input[k] * current_weights[k];
        }
        sum += biases[out_idx];
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// FC Backward Kernel
// Calculates gradients w.r.t FC input, weights, and biases
// Uses atomicAdd for potential race conditions in gradient accumulation.
__global__ void fc_backward_kernel(float* delta_in,         // Output [batch_size x in_features]
                                   float* grad_weights,     // Output [out_features x in_features]
                                   float* grad_biases,      // Output [out_features]
                                   const float* delta_out,  // Input [batch_size x out_features]
                                   const float* activation_in, // Input [batch_size x in_features]
                                   const float* weights,    // Input [out_features x in_features]
                                   int batch_size,
                                   int in_features,
                                   int out_features)
{
    // --- Calculate delta_in = delta_out * W^T ---
    // Each thread calculates one element of delta_in
    int batch_idx_in = blockIdx.y; // Map y-dim of grid to batch
    int in_idx = blockIdx.x * blockDim.x + threadIdx.x; // Map x-dim of grid/block to in_features

    if (batch_idx_in < batch_size && in_idx < in_features) {
        float sum_in = 0.0f;
        for (int k = 0; k < out_features; ++k) {
            // delta_out[batch_idx_in * out_features + k]
            // weights[k * in_features + in_idx]  (accessing W^T implicitly)
            sum_in += delta_out[batch_idx_in * out_features + k] * weights[k * in_features + in_idx];
        }
        delta_in[batch_idx_in * in_features + in_idx] = sum_in;
    }

    // --- Calculate grad_weights = activation_in^T * delta_out ---
    // Potential Race Condition: Multiple batch samples contribute to the same weight gradient.
    // Each thread calculates contribution for one weight element from one batch sample.
    // Use separate thread mapping or recalculate indices if needed.
    // Let's map threads to weight matrix elements [out_idx][in_idx]
    int out_idx_w = blockIdx.y * blockDim.y + threadIdx.y; // Map grid/block y to out_features
    int in_idx_w = blockIdx.x * blockDim.x + threadIdx.x;   // Map grid/block x to in_features

    if (out_idx_w < out_features && in_idx_w < in_features) {
        float sum_w_grad = 0.0f;
        for (int n = 0; n < batch_size; ++n) {
            // activation_in[n * in_features + in_idx_w] (Accessing activation_in^T implicitly)
            // delta_out[n * out_features + out_idx_w]
            sum_w_grad += activation_in[n * in_features + in_idx_w] * delta_out[n * out_features + out_idx_w];
        }
        // --- RACE CONDITION HERE ---
        // Multiple threads (from different blocks if grid > 1x1) could write to the same grad_weights element.
        // Using atomicAdd to safely accumulate. Assumes grad_weights was zeroed out before kernel launch.
         atomicAdd(&grad_weights[out_idx_w * in_features + in_idx_w], sum_w_grad);
        // If only one block calculates all weight gradients, sync is enough before writing.
        // But usually, we parallelize over weights too.
    }


    // --- Calculate grad_biases = sum(delta_out, axis=0) ---
    // Potential Race Condition: Multiple batch samples contribute to the same bias gradient.
    // Map threads to output features (bias index)
    int out_idx_b = blockIdx.x * blockDim.x + threadIdx.x; // Use 1D grid/block mapping for biases

    if (out_idx_b < out_features) {
        float sum_b_grad = 0.0f;
        for (int n = 0; n < batch_size; ++n) {
            sum_b_grad += delta_out[n * out_features + out_idx_b];
        }
        // --- RACE CONDITION HERE ---
        // Only one thread calculates the full sum for a given bias index if using 1D mapping *exactly* matching out_features.
        // If blockDim.x < out_features and gridDim.x > 1, atomics are needed.
        // For simplicity assuming blockDim.x=out_features, gridDim.x=1 or use atomicAdd
         atomicAdd(&grad_biases[out_idx_b], sum_b_grad);
        // grad_biases[out_idx_b] = sum_b_grad; // Only safe if exactly one thread computes this index.
    }
}


// Softmax Forward (Numerically Stable)
// Uses shared memory for reduction within a block (finding max, calculating sum)
// Assumes blockDim.x is >= num_classes and <= 1024 (max threads/block)
// Assumes gridDim.x = batch_size
__global__ void softmax_kernel(float* output, const float* input, int batch_size, int num_classes) {
    extern __shared__ float s_data[]; // Shared memory for intermediate results

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x; // Thread index within the block (0 to blockDim.x-1)

    // Load input for this sample into shared memory
    // Find max value in the input vector for numerical stability
    float thread_max_val = -FLT_MAX;
    if (tid < num_classes) {
        s_data[tid] = input[batch_idx * num_classes + tid];
        thread_max_val = s_data[tid];
    } else {
        s_data[tid] = -FLT_MAX; // Initialize padding threads if blockDim > num_classes
    }
     __syncthreads(); // Ensure all data is loaded

    // Parallel reduction in shared memory to find the max value
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] = max(s_data[tid], s_data[tid + stride]);
        }
         __syncthreads();
    }
    float max_val = s_data[0]; // Max value is now in s_data[0]
     __syncthreads(); // Ensure all threads have read max_val before overwriting s_data


    // Calculate exp(x - max_val) and sum
    float thread_exp_sum = 0.0f;
    if (tid < num_classes) {
        // Reload original value or recalculate if shared mem was reused
        float val = input[batch_idx * num_classes + tid];
        s_data[tid] = expf(val - max_val); // Use expf for float
        thread_exp_sum = s_data[tid];
    } else {
         s_data[tid] = 0.0f; // Padding threads
    }
     __syncthreads(); // Ensure all exp values are calculated

    // Parallel reduction in shared memory to find the sum of exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
         __syncthreads();
    }
    float sum_exp = s_data[0]; // Sum is now in s_data[0]
     __syncthreads(); // Ensure all threads have read sum_exp


    // Normalize to get probabilities
    if (tid < num_classes) {
        // Reload exp value or recalculate
        float exp_val = expf(input[batch_idx * num_classes + tid] - max_val);
        output[batch_idx * num_classes + tid] = exp_val / (sum_exp + 1e-8f); // Add epsilon for stability
    }
}


// Calculate Cross-Entropy Loss and Softmax Gradient
// Note: Gradient is calculated w.r.t the *input* to the softmax layer.
__global__ void softmax_cross_entropy_loss_backward_kernel(float* loss, // Output loss (single value using atomicAdd)
                                                           float* delta, // Output delta [batch_size x num_classes]
                                                           const float* softmax_output, // Input [batch_size x num_classes]
                                                           const int*   true_labels,    // Input [batch_size]
                                                           int batch_size,
                                                           int num_classes)
{
    int batch_idx = blockIdx.x; // Map blocks to batch samples
    int class_idx = threadIdx.x; // Map threads to classes

    if (batch_idx < batch_size && class_idx < num_classes) {
        int label = true_labels[batch_idx];
        int current_idx = batch_idx * num_classes + class_idx;

        float output_prob = softmax_output[current_idx];

        // Calculate gradient: delta = output - target
        // Target is 1 for the true class, 0 otherwise
        float target = (class_idx == label) ? 1.0f : 0.0f;
        delta[current_idx] = output_prob - target;

        // Calculate loss contribution for this sample (only thread for true class does this)
        if (class_idx == label) {
             float sample_loss = -logf(max(output_prob, 1e-8f)); // Use logf, add epsilon
             // --- RACE CONDITION for global loss ---
             // Atomically add this sample's loss to the total batch loss (assume loss[0] stores it)
             atomicAdd(loss, sample_loss);
        }
    }
}


// Simple SGD Weight Update Kernel
__global__ void sgd_update_kernel(float* weights,
                                  float* biases,
                                  const float* grad_weights,
                                  const float* grad_biases,
                                  float learning_rate,
                                  size_t num_weights,
                                  size_t num_biases) {
    // Update weights
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_idx < num_weights) {
        weights[weight_idx] -= learning_rate * grad_weights[weight_idx];
    }

    // Update biases (could use a separate kernel launch or combine carefully)
    // If blockDim.x maps potentially beyond num_biases for weight update,
    // we need separate index calculation or a separate launch.
    // Assuming biases are few, run this part only by first few threads/blocks.
     int bias_idx = blockIdx.x * blockDim.x + threadIdx.x; // Re-using index for simplicity
     if (bias_idx < num_biases) {
         // Ensure this thread index is valid for biases, maybe use threadIdx.x directly if blockDim.x >= num_biases
         biases[bias_idx] -= learning_rate * grad_biases[bias_idx];
     }
}


// --- Placeholder Kernel Implementations ---

__global__ void conv_forward_kernel(float* output, const float* input, const float* weights, const float* biases,
                                    int batch_size, int in_h, int in_w, int in_c,
                                    int out_h, int out_w, int out_c,
                                    int kernel_size, int stride, int padding) {
    // Placeholder: A real implementation would involve:
    // 1. Complex indexing for input, weights, output.
    // 2. Mapping threads/blocks to output elements (batch, out_h, out_w, out_c).
    // 3. Using shared memory to cache input tiles and potentially filter weights.
    // 4. Loops over input channels and kernel dimensions.
    // 5. __syncthreads() for shared memory coordination.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) { // Just print a message once to show it was called
         // printf("conv_forward_kernel placeholder called.\n");
    }
     // Actual computation missing
}

__global__ void pool_forward_kernel(float* output, int* max_indices, const float* input,
                                    int batch_size, int in_h, int in_w, int in_c,
                                    int out_h, int out_w, int pool_size, int stride) {
    // Placeholder: A real implementation would involve:
    // 1. Mapping threads/blocks to output elements (batch, out_h, out_w, in_c).
    // 2. Looping through the pooling window (pool_size x pool_size) in the input.
    // 3. Finding the maximum value within the window.
    // 4. Storing the maximum value in 'output'.
    // 5. Storing the *index* (relative to input) of the max value in 'max_indices'.
    // 6. Shared memory might be used for input caching.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx == 0) { // Just print a message once
         // printf("pool_forward_kernel placeholder called.\n");
     }
    // Actual computation missing
}


__global__ void conv_backward_kernel(float* delta_in, float* grad_weights, float* grad_biases,
                                     const float* delta_out, const float* activation_in, const float* weights,
                                     int batch_size, int in_h, int in_w, int in_c,
                                     int out_h, int out_w, int out_c,
                                     int kernel_size, int stride, int padding) {
    // Placeholder: Extremely complex. Involves:
    // 1. Calculating delta_in: Requires convolution of delta_out with rotated filters (W_rot180).
    // 2. Calculating grad_weights: Requires convolution of activation_in with delta_out.
    // 3. Calculating grad_biases: Summing delta_out over spatial dimensions (h, w) for each filter/channel.
    // 4. Extensive use of shared memory and potentially atomicAdd for gradient accumulation.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx == 0) { // Just print a message once
          // printf("conv_backward_kernel placeholder called.\n");
     }
     // Actual computation missing
}

__global__ void pool_backward_kernel(float* delta_in, const float* delta_out, const int* max_indices,
                                     int batch_size, int in_h, int in_w, int in_c,
                                     int out_h, int out_w, int pool_size, int stride) {
    // Placeholder: Involves:
    // 1. Mapping threads perhaps to input elements or output elements.
    // 2. Reading the corresponding delta_out value.
    // 3. Reading the max_index saved during the forward pass.
    // 4. Adding (atomically, due to potential overlaps if stride < pool_size) the delta_out value
    //    to the delta_in element at the max_index location. delta_in needs to be zeroed beforehand.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx == 0) { // Just print a message once
         // printf("pool_backward_kernel placeholder called.\n");
     }
    // Actual computation missing - REQUIRES atomicAdd potentially on delta_in
}