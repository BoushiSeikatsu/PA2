#ifndef CNN_H
#define CNN_H

#include <vector>
#include <cstddef> // For size_t

// Define Layer Types
typedef enum {
    LAYER_INPUT,
    LAYER_CONV,
    LAYER_RELU,
    LAYER_POOL,
    LAYER_FLATTEN,
    LAYER_FC,
    LAYER_SOFTMAX // Often combined with loss
} LayerType;

// Basic Layer Parameters (Extend as needed)
typedef struct {
    LayerType type;
    int input_height, input_width, input_channels;
    int output_height, output_width, output_channels;

    // Conv params
    int num_filters;
    int kernel_size;
    int stride;
    int padding; // Assuming padding=0 for simplicity here

    // Pool params
    int pool_size;
    // pool_stride assumed same as pool_size for simplicity

    // FC params
    int num_neurons;

    // --- Calculated offsets/sizes within global buffers ---
    size_t weight_offset;
    size_t bias_offset;
    size_t weight_size; // Number of float elements
    size_t bias_size;   // Number of float elements
    size_t output_activation_size; // Size per sample (h*w*c or neurons)

} LayerParams;


// Network Structure (primarily Host-side metadata)
typedef struct {
    int num_layers;
    std::vector<LayerParams> layers; // More flexible than fixed array

    size_t total_weights_size; // Total float elements
    size_t total_biases_size;
    size_t max_activation_elements_per_sample; // Max h*w*c across layers
    size_t total_activation_buffer_size; // max_activation_elements_per_sample * batch_size
    size_t input_elements_per_sample; // h*w*c for input layer
    size_t output_elements_per_sample; // num_classes for output layer

    int batch_size;
    int input_height, input_width, input_channels;
    int num_classes;

    // --- Device Pointers (allocated on Host, point to Device memory) ---
    float* d_weights;
    float* d_biases;
    float* d_activations[2]; // Ping-pong buffers
    float* d_delta_activations[2]; // Ping-pong buffers for backprop deltas
    float* d_gradients_weights;
    float* d_gradients_biases;
    float* d_input_batch;
    int*   d_labels_batch;
    float* d_output; // Final network output (after softmax)
    float* d_loss; // Per-sample loss or single aggregated loss
    int*   d_pool_indices; // Aux buffer to store max indices from pooling (for backprop)
                           // Size needs calculation based on pooling layers

} Network;

// Function to build the network structure
void build_network(Network& net, int batch_size);

// Function to allocate memory based on network structure
void allocate_network_memory(Network& net);

// Function to free memory
void free_network_memory(Network& net);

#endif // CNN_H