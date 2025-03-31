#include "cnn.h"
#include "utils.h"
#include <stdexcept> // For runtime_error
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::max

// Helper to calculate output size for Conv/Pool
void calculate_output_dims(int in_h, int in_w, int kernel_size, int stride, int padding, int& out_h, int& out_w) {
    // Simple calculation, assumes symmetric kernel/stride/padding
    out_h = (in_h - kernel_size + 2 * padding) / stride + 1;
    out_w = (in_w - kernel_size + 2 * padding) / stride + 1;
}

// Define the specific network architecture
void build_network(Network& net, int batch_size) {
    net.batch_size = batch_size;
    net.input_height = 28;
    net.input_width = 28;
    net.input_channels = 1;
    net.num_classes = 10;
    net.layers.clear();

    int h = net.input_height;
    int w = net.input_width;
    int c = net.input_channels;
    net.input_elements_per_sample = h * w * c;
    net.total_weights_size = 0;
    net.total_biases_size = 0;
    net.max_activation_elements_per_sample = net.input_elements_per_sample;

    // Layer 0: Implicit Input Layer
    LayerParams input_layer = {};
    input_layer.type = LAYER_INPUT;
    input_layer.output_height = h;
    input_layer.output_width = w;
    input_layer.output_channels = c;
    input_layer.output_activation_size = h * w * c;
    net.layers.push_back(input_layer);


    // Layer 1: Conv1
    LayerParams conv1 = {};
    conv1.type = LAYER_CONV;
    conv1.input_height = h; conv1.input_width = w; conv1.input_channels = c;
    conv1.num_filters = 6;
    conv1.kernel_size = 5; conv1.stride = 1; conv1.padding = 0;
    calculate_output_dims(h, w, conv1.kernel_size, conv1.stride, conv1.padding, conv1.output_height, conv1.output_width);
    conv1.output_channels = conv1.num_filters;
    conv1.weight_offset = net.total_weights_size;
    conv1.weight_size = conv1.num_filters * c * conv1.kernel_size * conv1.kernel_size;
    conv1.bias_offset = net.total_biases_size;
    conv1.bias_size = conv1.num_filters;
    conv1.output_activation_size = conv1.output_height * conv1.output_width * conv1.output_channels;
    net.total_weights_size += conv1.weight_size;
    net.total_biases_size += conv1.bias_size;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, conv1.output_activation_size);
    net.layers.push_back(conv1);
    h = conv1.output_height; w = conv1.output_width; c = conv1.output_channels;

    // Layer 2: ReLU1
    LayerParams relu1 = {};
    relu1.type = LAYER_RELU;
    relu1.input_height = h; relu1.input_width = w; relu1.input_channels = c;
    relu1.output_height = h; relu1.output_width = w; relu1.output_channels = c;
    relu1.output_activation_size = h * w * c;
    // No weights/biases
    relu1.weight_size = 0; relu1.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, relu1.output_activation_size);
    net.layers.push_back(relu1);

    // Layer 3: Pool1
    LayerParams pool1 = {};
    pool1.type = LAYER_POOL;
    pool1.input_height = h; pool1.input_width = w; pool1.input_channels = c;
    pool1.pool_size = 2; pool1.stride = 2; // Assuming stride=pool_size
    calculate_output_dims(h, w, pool1.pool_size, pool1.stride, 0, pool1.output_height, pool1.output_width);
    pool1.output_channels = c;
    pool1.output_activation_size = pool1.output_height * pool1.output_width * pool1.output_channels;
    pool1.weight_size = 0; pool1.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, pool1.output_activation_size);
    net.layers.push_back(pool1);
    h = pool1.output_height; w = pool1.output_width; c = pool1.output_channels;

    // Layer 4: Conv2
    LayerParams conv2 = {};
    conv2.type = LAYER_CONV;
    conv2.input_height = h; conv2.input_width = w; conv2.input_channels = c;
    conv2.num_filters = 16;
    conv2.kernel_size = 5; conv2.stride = 1; conv2.padding = 0;
    calculate_output_dims(h, w, conv2.kernel_size, conv2.stride, conv2.padding, conv2.output_height, conv2.output_width);
    conv2.output_channels = conv2.num_filters;
    conv2.weight_offset = net.total_weights_size;
    conv2.weight_size = conv2.num_filters * c * conv2.kernel_size * conv2.kernel_size;
    conv2.bias_offset = net.total_biases_size;
    conv2.bias_size = conv2.num_filters;
    conv2.output_activation_size = conv2.output_height * conv2.output_width * conv2.output_channels;
    net.total_weights_size += conv2.weight_size;
    net.total_biases_size += conv2.bias_size;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, conv2.output_activation_size);
    net.layers.push_back(conv2);
    h = conv2.output_height; w = conv2.output_width; c = conv2.output_channels;

    // Layer 5: ReLU2
    LayerParams relu2 = {};
    relu2.type = LAYER_RELU;
    relu2.input_height = h; relu2.input_width = w; relu2.input_channels = c;
    relu2.output_height = h; relu2.output_width = w; relu2.output_channels = c;
    relu2.output_activation_size = h * w * c;
    relu2.weight_size = 0; relu2.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, relu2.output_activation_size);
    net.layers.push_back(relu2);

    // Layer 6: Pool2
    LayerParams pool2 = {};
    pool2.type = LAYER_POOL;
    pool2.input_height = h; pool2.input_width = w; pool2.input_channels = c;
    pool2.pool_size = 2; pool2.stride = 2;
    calculate_output_dims(h, w, pool2.pool_size, pool2.stride, 0, pool2.output_height, pool2.output_width);
    pool2.output_channels = c;
    pool2.output_activation_size = pool2.output_height * pool2.output_width * pool2.output_channels;
    pool2.weight_size = 0; pool2.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, pool2.output_activation_size);
    net.layers.push_back(pool2);
    h = pool2.output_height; w = pool2.output_width; c = pool2.output_channels;

    // Layer 7: Flatten
    LayerParams flatten = {};
    flatten.type = LAYER_FLATTEN;
    flatten.input_height = h; flatten.input_width = w; flatten.input_channels = c;
    flatten.output_height = 1; flatten.output_width = 1; flatten.output_channels = h * w * c; // Flattened size
    flatten.output_activation_size = flatten.output_channels;
    flatten.weight_size = 0; flatten.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, flatten.output_activation_size);
    net.layers.push_back(flatten);
    int flattened_size = flatten.output_channels; // Use channels as size for FC layers

    // Layer 8: FC1
    LayerParams fc1 = {};
    fc1.type = LAYER_FC;
    fc1.input_channels = flattened_size; // Input size
    fc1.num_neurons = 120;
    fc1.output_channels = fc1.num_neurons; // Output size
    fc1.output_activation_size = fc1.output_channels;
    fc1.weight_offset = net.total_weights_size;
    fc1.weight_size = fc1.output_channels * fc1.input_channels;
    fc1.bias_offset = net.total_biases_size;
    fc1.bias_size = fc1.output_channels;
    net.total_weights_size += fc1.weight_size;
    net.total_biases_size += fc1.bias_size;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, fc1.output_activation_size);
    net.layers.push_back(fc1);
    int current_fc_size = fc1.output_channels;

    // Layer 9: ReLU3
    LayerParams relu3 = {};
    relu3.type = LAYER_RELU;
    relu3.input_channels = current_fc_size;
    relu3.output_channels = current_fc_size;
    relu3.output_activation_size = current_fc_size;
    relu3.weight_size = 0; relu3.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, relu3.output_activation_size);
    net.layers.push_back(relu3);

    // Layer 10: FC2
    LayerParams fc2 = {};
    fc2.type = LAYER_FC;
    fc2.input_channels = current_fc_size;
    fc2.num_neurons = 84;
    fc2.output_channels = fc2.num_neurons;
    fc2.output_activation_size = fc2.output_channels;
    fc2.weight_offset = net.total_weights_size;
    fc2.weight_size = fc2.output_channels * fc2.input_channels;
    fc2.bias_offset = net.total_biases_size;
    fc2.bias_size = fc2.output_channels;
    net.total_weights_size += fc2.weight_size;
    net.total_biases_size += fc2.bias_size;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, fc2.output_activation_size);
    net.layers.push_back(fc2);
    current_fc_size = fc2.output_channels;

    // Layer 11: ReLU4 (Optional - sometimes omitted before final FC)
    LayerParams relu4 = {};
    relu4.type = LAYER_RELU;
    relu4.input_channels = current_fc_size;
    relu4.output_channels = current_fc_size;
    relu4.output_activation_size = current_fc_size;
    relu4.weight_size = 0; relu4.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, relu4.output_activation_size);
    net.layers.push_back(relu4);


    // Layer 12: FC3 (Output Layer)
    LayerParams fc3 = {};
    fc3.type = LAYER_FC;
    fc3.input_channels = current_fc_size;
    fc3.num_neurons = net.num_classes;
    fc3.output_channels = fc3.num_neurons;
    fc3.output_activation_size = fc3.output_channels;
    fc3.weight_offset = net.total_weights_size;
    fc3.weight_size = fc3.output_channels * fc3.input_channels;
    fc3.bias_offset = net.total_biases_size;
    fc3.bias_size = fc3.output_channels;
    net.total_weights_size += fc3.weight_size;
    net.total_biases_size += fc3.bias_size;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, fc3.output_activation_size);
    net.layers.push_back(fc3);

    // Layer 13: Softmax (handled separately or combined with loss)
    LayerParams softmax = {};
    softmax.type = LAYER_SOFTMAX;
    softmax.input_channels = net.num_classes;
    softmax.output_channels = net.num_classes;
    softmax.output_activation_size = net.num_classes;
    softmax.weight_size = 0; softmax.bias_size = 0;
    net.max_activation_elements_per_sample = std::max(net.max_activation_elements_per_sample, softmax.output_activation_size);
    net.layers.push_back(softmax);


    net.num_layers = net.layers.size();
    net.output_elements_per_sample = net.num_classes;
    net.total_activation_buffer_size = net.max_activation_elements_per_sample * net.batch_size;

    // Calculate size for pooling indices buffer (sum over pool layers)
    size_t total_pool_indices = 0;
    for(const auto& layer : net.layers) {
        if (layer.type == LAYER_POOL) {
            total_pool_indices += layer.output_activation_size; // Indices needed for each output element
        }
    }
    net.d_pool_indices = nullptr; // Allocate this in allocate_network_memory

    printf("Network Built:\n");
    printf("  Total Layers: %d\n", net.num_layers);
    printf("  Total Weights: %zu\n", net.total_weights_size);
    printf("  Total Biases: %zu\n", net.total_biases_size);
    printf("  Max Activation Size / Sample: %zu\n", net.max_activation_elements_per_sample);
    printf("  Activation Buffer Size (Batch %d): %zu elements (%zu MB)\n",
           net.batch_size, net.total_activation_buffer_size,
           net.total_activation_buffer_size * sizeof(float) / (1024*1024));
    printf("  Pooling Indices Buffer Size (Batch %d): %zu elements (%zu MB)\n",
            net.batch_size, total_pool_indices * net.batch_size,
             total_pool_indices * net.batch_size * sizeof(int) / (1024*1024) );

}

void allocate_network_memory(Network& net) {
    printf("Allocating GPU Memory...\n");
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_weights, net.total_weights_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_biases, net.total_biases_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_gradients_weights, net.total_weights_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_gradients_biases, net.total_biases_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&net.d_activations[0], net.total_activation_buffer_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_activations[1], net.total_activation_buffer_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_delta_activations[0], net.total_activation_buffer_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_delta_activations[1], net.total_activation_buffer_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&net.d_input_batch, net.input_elements_per_sample * net.batch_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_labels_batch, net.batch_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_output, net.output_elements_per_sample * net.batch_size * sizeof(float)));

    // Allocate space for loss (e.g., one float for aggregated batch loss)
    CHECK_CUDA_ERROR(cudaMalloc(&net.d_loss, 1 * sizeof(float)));

    // Allocate pooling indices buffer if needed
    size_t total_pool_indices = 0;
     for(const auto& layer : net.layers) {
         if (layer.type == LAYER_POOL) {
             total_pool_indices += layer.output_activation_size;
         }
     }
     if (total_pool_indices > 0) {
         CHECK_CUDA_ERROR(cudaMalloc(&net.d_pool_indices, total_pool_indices * net.batch_size * sizeof(int)));
     } else {
         net.d_pool_indices = nullptr;
     }


    printf("GPU Memory Allocated.\n");
}

void free_network_memory(Network& net) {
    printf("Freeing GPU Memory...\n");
    CHECK_CUDA_ERROR(cudaFree(net.d_weights));
    CHECK_CUDA_ERROR(cudaFree(net.d_biases));
    CHECK_CUDA_ERROR(cudaFree(net.d_gradients_weights));
    CHECK_CUDA_ERROR(cudaFree(net.d_gradients_biases));
    CHECK_CUDA_ERROR(cudaFree(net.d_activations[0]));
    CHECK_CUDA_ERROR(cudaFree(net.d_activations[1]));
    CHECK_CUDA_ERROR(cudaFree(net.d_delta_activations[0]));
    CHECK_CUDA_ERROR(cudaFree(net.d_delta_activations[1]));
    CHECK_CUDA_ERROR(cudaFree(net.d_input_batch));
    CHECK_CUDA_ERROR(cudaFree(net.d_labels_batch));
    CHECK_CUDA_ERROR(cudaFree(net.d_output));
    CHECK_CUDA_ERROR(cudaFree(net.d_loss));
    if (net.d_pool_indices) {
        CHECK_CUDA_ERROR(cudaFree(net.d_pool_indices));
    }
    printf("GPU Memory Freed.\n");
}

// Placeholder / Simple weight initialization
void initialize_weights_xavier(float* weights, int in_features, int out_features) {
     float range = sqrtf(6.0f / (in_features + out_features));
     for (int i = 0; i < out_features * in_features; ++i) {
         weights[i] = ((float)rand() / RAND_MAX) * 2.0f * range - range;
     }
}

void initialize_biases(float* biases, int num_features) {
     for (int i = 0; i < num_features; ++i) {
         biases[i] = 0.0f; // Often initialized to zero
     }
}