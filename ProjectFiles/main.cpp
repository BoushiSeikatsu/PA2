#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // For timing
#include <numeric> // std::iota
#include <random>  // std::shuffle
#include <algorithm> // std::shuffle

#include "utils.h"
#include "cnn.h"
#include "kernels.h"

// --- Forward Pass Function ---
void forward_pass(Network& net) {
    int current_buffer = 0;
    float* current_input_ptr = net.d_input_batch; // Start with batch input

    for (int i = 1; i < net.num_layers; ++i) { // Start from layer 1 (skip implicit input)
        LayerParams& layer = net.layers[i];
        LayerParams& prev_layer = net.layers[i-1];
        float* current_output_ptr = net.d_activations[current_buffer];

        // Determine input size for this layer
        size_t input_size_elements = (size_t)prev_layer.output_activation_size * net.batch_size;
         if (layer.type == LAYER_FLATTEN) input_size_elements = (size_t)prev_layer.output_height* prev_layer.output_width*prev_layer.output_channels* net.batch_size;


        // Determine output size for this layer
        size_t output_size_elements = (size_t)layer.output_activation_size * net.batch_size;
         if (layer.type == LAYER_FLATTEN) output_size_elements = input_size_elements; // Flatten just reshapes


        // Pointers to weights and biases for the current layer
        float* layer_weights = net.d_weights + layer.weight_offset;
        float* layer_biases = net.d_biases + layer.bias_offset;

        // --- Kernel Launch based on Layer Type ---
        dim3 blockDim, gridDim;
        size_t N; // Total elements to process

        switch (layer.type) {
            case LAYER_CONV:
                // Complex grid/block setup needed based on output dimensions
                 blockDim = dim3(16, 16); // Example
                 gridDim = dim3((layer.output_width + blockDim.x - 1) / blockDim.x,
                                (layer.output_height + blockDim.y - 1) / blockDim.y,
                                net.batch_size * layer.output_channels); // Example mapping
                conv_forward_kernel<<<gridDim, blockDim>>>(
                    current_output_ptr, current_input_ptr, layer_weights, layer_biases,
                    net.batch_size, prev_layer.output_height, prev_layer.output_width, prev_layer.output_channels,
                    layer.output_height, layer.output_width, layer.output_channels,
                    layer.kernel_size, layer.stride, layer.padding);
                CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_RELU:
                N = output_size_elements;
                blockDim = dim3(256);
                gridDim = dim3((N + blockDim.x - 1) / blockDim.x);
                relu_forward_kernel<<<gridDim, blockDim>>>(current_output_ptr, current_input_ptr, N);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_POOL:
                 // Complex grid/block setup
                 blockDim = dim3(16, 16); // Example
                 gridDim = dim3((layer.output_width + blockDim.x - 1) / blockDim.x,
                                (layer.output_height + blockDim.y - 1) / blockDim.y,
                                net.batch_size * layer.output_channels); // Example
                 // Need to calculate offset into d_pool_indices buffer
                 // For simplicity, assume it's managed correctly outside this snippet
                 pool_forward_kernel<<<gridDim, blockDim>>>(
                    current_output_ptr, net.d_pool_indices, /* Pass correct offset */
                    current_input_ptr,
                    net.batch_size, prev_layer.output_height, prev_layer.output_width, prev_layer.output_channels,
                    layer.output_height, layer.output_width,
                    layer.pool_size, layer.stride);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_FLATTEN:
                // Flatten is just a logical reshape, copy data if necessary or manage pointers
                // If buffers are compatible, we might just reuse the pointer
                // For simplicity, assume output buffer is ready and just copy
                 if (current_input_ptr != current_output_ptr) {
                     CHECK_CUDA_ERROR(cudaMemcpy(current_output_ptr, current_input_ptr, output_size_elements * sizeof(float), cudaMemcpyDeviceToDevice));
                 }
                break;

             case LAYER_FC:
                 // GridDim maps blocks to batch samples and output features (partially)
                 blockDim = dim3(256); // Threads map to output features within a block
                 gridDim = dim3((layer.output_channels + blockDim.x - 1) / blockDim.x, net.batch_size);
                 fc_forward_kernel<<<gridDim, blockDim>>>(
                     current_output_ptr, current_input_ptr, layer_weights, layer_biases,
                     net.batch_size, prev_layer.output_channels, layer.output_channels);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                 break;

            case LAYER_SOFTMAX:
                 blockDim = dim3(net.num_classes <= 1024 ? net.num_classes : 1024); // Use enough threads for reduction
                 gridDim = dim3(net.batch_size); // One block per sample
                 size_t shared_mem_size = blockDim.x * sizeof(float);
                 softmax_kernel<<<gridDim, blockDim, shared_mem_size>>>(
                     net.d_output, // Write final output to dedicated buffer
                     current_input_ptr,
                     net.batch_size, net.num_classes);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                 // After softmax, the final output is in net.d_output
                 current_output_ptr = net.d_output; // Update pointer for loss calc if needed
                 break;

             default:
                 // Skip unknown layers
                 break;

        }

        // Prepare for next layer
        current_input_ptr = current_output_ptr; // Output of this layer is input to next
         if (layer.type != LAYER_SOFTMAX) { // Softmax writes to specific buffer
             current_buffer = 1 - current_buffer; // Switch ping-pong buffer
         }

        // Optional: Synchronize after each layer for debugging
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

// --- Backward Pass Function ---
void backward_pass(Network& net) {
    int current_delta_buffer = 0; // Start writing initial delta here
    float* current_delta_out_ptr = nullptr; // Gradient w.r.t output of the current layer being processed
    float* current_delta_in_ptr = nullptr;  // Gradient w.r.t input of the current layer (output of kernel)

    // 1. Compute Initial Delta (Gradient w.r.t Softmax Input) & Loss
    CHECK_CUDA_ERROR(cudaMemset(net.d_loss, 0, sizeof(float))); // Zero out loss accumulator
    dim3 blockDim_loss = dim3(net.num_classes <= 1024 ? net.num_classes : 1024);
    dim3 gridDim_loss = dim3(net.batch_size);
    softmax_cross_entropy_loss_backward_kernel<<<gridDim_loss, blockDim_loss>>>(
        net.d_loss,
        net.d_delta_activations[current_delta_buffer], // Write initial delta here
        net.d_output, // Read softmax output
        net.d_labels_batch,
        net.batch_size, net.num_classes);
    CHECK_CUDA_ERROR(cudaGetLastError());

    current_delta_out_ptr = net.d_delta_activations[current_delta_buffer]; // This is delta w.r.t output of layer N-1

    // 2. Iterate backwards through layers (skip Softmax layer, start from layer before it)
    for (int i = net.num_layers - 2; i >= 1; --i) {
        LayerParams& layer = net.layers[i];
        LayerParams& prev_layer = net.layers[i-1]; // Layer whose output feeds into current layer 'i'

        // Determine which activation buffer holds the input to *this* layer during forward pass
        // This requires tracking the ping-pong buffers used in forward pass, or recalculating.
        // Simplification: Assume we know the correct forward activation buffer index 'fwd_buffer_idx' for layer 'i's input.
        // This is tricky to manage correctly without storing the buffer index per layer during forward pass.
        // Let's assume net.d_activations[?] contains activation_in for layer 'i'.
        // We need the activations *before* the non-linearity for ReLU backward.
        // This structure needs refinement for correct activation buffering/retrieval in backprop.
        // TODO: Fix activation buffer selection for backprop (requires more state tracking).
        // Using placeholder: assume net.d_activations[0] holds the relevant input activations
        float* activation_in_ptr = net.d_activations[0]; // *** Needs correction ***
        if (layer.type == LAYER_RELU) {
            // Need activations *before* ReLU, which are output of layer i-1
             // activation_in_ptr = ... get buffer for prev_layer.output ...
             // This highlights the complexity of buffer management.
             // Let's assume for now the correct ptr is available.
        } else if (layer.type == LAYER_FC || layer.type == LAYER_CONV) {
             // Need activations that were input to this layer (output of prev_layer)
             // activation_in_ptr = ... get buffer for prev_layer.output ...
        }


        // Output buffer for the gradient w.r.t the input of this layer
        current_delta_in_ptr = net.d_delta_activations[1 - current_delta_buffer];

        // Pointers to weights/biases and their gradients
        float* layer_weights = net.d_weights + layer.weight_offset;
        float* layer_grad_weights = net.d_gradients_weights + layer.weight_offset;
        float* layer_grad_biases = net.d_gradients_biases + layer.bias_offset;

        // --- Kernel Launch based on Layer Type ---
        dim3 blockDim, gridDim;
        size_t N;

         // *** IMPORTANT: Zero out gradient buffers before accumulating ***
         if (layer.weight_size > 0) {
             CHECK_CUDA_ERROR(cudaMemset(layer_grad_weights, 0, layer.weight_size * sizeof(float)));
         }
         if (layer.bias_size > 0) {
             CHECK_CUDA_ERROR(cudaMemset(layer_grad_biases, 0, layer.bias_size * sizeof(float)));
         }
        // We also need to zero out delta_in buffer if using atomicAdd in subsequent layer backprop (like pooling)
        // CHECK_CUDA_ERROR(cudaMemset(current_delta_in_ptr, 0, /* size */));


        switch (layer.type) {
            case LAYER_CONV:
                // Complex grid/block setup
                conv_backward_kernel<<<gridDim, blockDim>>>( /* Placeholder call */
                    current_delta_in_ptr, layer_grad_weights, layer_grad_biases,
                    current_delta_out_ptr, activation_in_ptr /* Correct buffer needed */, layer_weights,
                    net.batch_size, prev_layer.output_height, prev_layer.output_width, prev_layer.output_channels,
                    layer.output_height, layer.output_width, layer.output_channels,
                    layer.kernel_size, layer.stride, layer.padding);
                CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_RELU:
                 N = (size_t)layer.output_activation_size * net.batch_size;
                 blockDim = dim3(256);
                 gridDim = dim3((N + blockDim.x - 1) / blockDim.x);
                 // Pass the activations *before* ReLU was applied (output of layer i-1)
                 // This requires careful buffer management from fwd pass. Using placeholder:
                 float* pre_relu_activations = net.d_activations[0]; // *** Needs correction ***
                 relu_backward_kernel<<<gridDim, blockDim>>>(
                     current_delta_in_ptr, // Output delta w.r.t ReLU input
                     current_delta_out_ptr, // Input delta w.r.t ReLU output
                     pre_relu_activations, // Input activations *before* ReLU
                     N);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_POOL:
                // Complex grid/block setup
                 // Needs max_indices from forward pass
                 // Needs zeroed delta_in buffer because pool_backward uses atomicAdd internally
                 CHECK_CUDA_ERROR(cudaMemset(current_delta_in_ptr, 0, (size_t)prev_layer.output_activation_size * net.batch_size * sizeof(float)));
                 pool_backward_kernel<<<gridDim, blockDim>>>( /* Placeholder call */
                     current_delta_in_ptr, current_delta_out_ptr,
                     net.d_pool_indices, /* Correct offset needed */
                     net.batch_size, prev_layer.output_height, prev_layer.output_width, prev_layer.output_channels,
                     layer.output_height, layer.output_width,
                     layer.pool_size, layer.stride);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                break;

            case LAYER_FLATTEN:
                // Flatten backward is just a reshape, copy delta if needed
                if (current_delta_out_ptr != current_delta_in_ptr) {
                     CHECK_CUDA_ERROR(cudaMemcpy(current_delta_in_ptr, current_delta_out_ptr, (size_t)layer.output_activation_size * net.batch_size * sizeof(float), cudaMemcpyDeviceToDevice));
                 }
                break;

            case LAYER_FC:
                 // Grid/Block Dim setup needs care, especially for gradient calculation
                 // Example setup for parallelizing over weight matrix and batch for grads:
                 blockDim = dim3(16, 16); // Example for weight grads
                 gridDim = dim3((prev_layer.output_channels + blockDim.x - 1) / blockDim.x,
                                (layer.output_channels + blockDim.y - 1) / blockDim.y);
                 // Need activations that were input to this layer (output of layer i-1)
                  float* fc_input_activations = net.d_activations[0]; // *** Needs correction ***

                 fc_backward_kernel<<<gridDim, blockDim>>>(
                     current_delta_in_ptr, layer_grad_weights, layer_grad_biases,
                     current_delta_out_ptr, fc_input_activations, layer_weights,
                     net.batch_size, prev_layer.output_channels, layer.output_channels);
                 CHECK_CUDA_ERROR(cudaGetLastError());
                 break;

             default:
                 // Skip layers without backward pass (like Input) or unknown
                 // If skipping, ensure delta is passed through correctly (e.g., copy)
                  if (current_delta_out_ptr != current_delta_in_ptr) {
                     CHECK_CUDA_ERROR(cudaMemcpy(current_delta_in_ptr, current_delta_out_ptr, /*size*/ sizeof(float), cudaMemcpyDeviceToDevice));
                 }
                 break;
        }

        // Prepare for next iteration (moving backward)
        current_delta_out_ptr = current_delta_in_ptr; // Delta calculated for this layer's input is the delta for the previous layer's output
        current_delta_buffer = 1 - current_delta_buffer; // Switch delta buffer

        // Optional: Synchronize after each layer for debugging
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

// --- Weight Update Function ---
void update_weights(Network& net, float learning_rate) {
    size_t total_params = net.total_weights_size + net.total_biases_size;
    if (total_params == 0) return;

    // Can use one kernel if indices are managed carefully, or separate kernels
    dim3 blockDim = dim3(256);
    dim3 gridDim_w = dim3((net.total_weights_size + blockDim.x - 1) / blockDim.x);
    // dim3 gridDim_b = dim3((net.total_biases_size + blockDim.x - 1) / blockDim.x);

    // Using single kernel launch, need careful indexing inside kernel if sizes differ greatly
    size_t max_size = std::max(net.total_weights_size, net.total_biases_size);
    dim3 gridDim = dim3((max_size + blockDim.x - 1) / blockDim.x);


    sgd_update_kernel<<<gridDim, blockDim>>>(
        net.d_weights, net.d_biases,
        net.d_gradients_weights, net.d_gradients_biases,
        learning_rate,
        net.total_weights_size, net.total_biases_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


// --- Main Function ---
int main() {
    // --- Config ---
    const int BATCH_SIZE = 128;
    const int EPOCHS = 10;
    const float LEARNING_RATE = 0.01f;
    const char* MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte"; // Replace with actual paths
    const char* MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte";
    // Add test image/label paths if doing evaluation

    // --- Load Data (Placeholder) ---
    std::vector<float> h_train_images;
    std::vector<int> h_train_labels;
    int num_train_images, img_height, img_width;
    // if (!load_mnist_images(MNIST_TRAIN_IMAGES, h_train_images, num_train_images, img_height, img_width)) return 1;
    // if (!load_mnist_labels(MNIST_TRAIN_LABELS, h_train_labels, num_train_images)) return 1;
    // Fake data for structure testing:
    num_train_images = 6000; // Smaller number for quick tests
    img_height = 28; img_width = 28;
    h_train_images.resize((size_t)num_train_images * img_height * img_width, 0.5f); // Fill with dummy value
    h_train_labels.resize(num_train_images);
    for(int i=0; i<num_train_images; ++i) h_train_labels[i] = i % 10; // Dummy labels
    printf("Loaded %d training samples.\n", num_train_images);


    // --- Build Network & Allocate Memory ---
    Network net = {};
    build_network(net, BATCH_SIZE);
    allocate_network_memory(net);

    // --- Initialize Weights & Biases on Host ---
    srand(42); // for reproducibility
    std::vector<float> h_weights(net.total_weights_size);
    std::vector<float> h_biases(net.total_biases_size);
    // Initialize layer by layer (example for FC layers) - Needs full implementation matching build_network
     for(const auto& layer : net.layers) {
         if (layer.type == LAYER_FC) {
             initialize_weights_xavier(h_weights.data() + layer.weight_offset, layer.input_channels, layer.output_channels);
             initialize_biases(h_biases.data() + layer.bias_offset, layer.output_channels);
         }
         // TODO: Add initialization for Conv layers
     }
     printf("Weights and biases initialized on host.\n");


    // --- Copy Initial Weights/Biases to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(net.d_weights, h_weights.data(), net.total_weights_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(net.d_biases, h_biases.data(), net.total_biases_size * sizeof(float), cudaMemcpyHostToDevice));
    printf("Initial weights and biases copied to GPU.\n");

    // --- Training Loop ---
    int num_batches = num_train_images / BATCH_SIZE;
    std::vector<int> indices(num_train_images);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., N-1
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());


    printf("\n--- Starting Training ---\n");
    auto start_total_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_epoch_time = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;

        std::shuffle(indices.begin(), indices.end(), rng); // Shuffle data indices

        for (int batch = 0; batch < num_batches; ++batch) {
            // 1. Prepare Batch Data on Host (using shuffled indices)
            std::vector<float> h_batch_images(BATCH_SIZE * net.input_elements_per_sample);
            std::vector<int> h_batch_labels(BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                int data_idx = indices[batch * BATCH_SIZE + i];
                // Copy image data (assuming h_train_images is flat)
                 std::copy(h_train_images.begin() + data_idx * net.input_elements_per_sample,
                           h_train_images.begin() + (data_idx + 1) * net.input_elements_per_sample,
                           h_batch_images.begin() + i * net.input_elements_per_sample);
                // Copy label
                h_batch_labels[i] = h_train_labels[data_idx];
            }

            // 2. Copy Batch to Device
            // Use cudaMemcpyAsync with streams for better performance
            CHECK_CUDA_ERROR(cudaMemcpy(net.d_input_batch, h_batch_images.data(), h_batch_images.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(net.d_labels_batch, h_batch_labels.data(), h_batch_labels.size() * sizeof(int), cudaMemcpyHostToDevice));

            // 3. Forward Pass
            forward_pass(net);

            // 4. Backward Pass (includes loss calculation)
            backward_pass(net);

            // 5. Update Weights
            update_weights(net, LEARNING_RATE);

            // 6. Copy Loss D->H & Accumulate (synchronizes)
            float batch_loss_gpu;
            CHECK_CUDA_ERROR(cudaMemcpy(&batch_loss_gpu, net.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_loss += batch_loss_gpu;

            // Optional: Sync explicitly if not copying loss, for accurate timing per batch
            // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            if ((batch + 1) % 100 == 0) { // Print progress
                 printf("  Epoch %d, Batch %d/%d, Avg Batch Loss: %.4f\n", epoch + 1, batch + 1, num_batches, epoch_loss / ((batch + 1)*BATCH_SIZE));
            }

        } // End Batch Loop

        auto end_epoch_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = end_epoch_time - start_epoch_time;

        printf("Epoch %d completed in %.3f s. Average Loss: %.4f\n",
               epoch + 1, epoch_duration.count(), epoch_loss / num_train_images);

    } // End Epoch Loop

    auto end_total_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total_time - start_total_time;
    printf("--- Training Finished in %.3f s ---\n", total_duration.count());


    // --- Copy Final Weights Back (Optional) ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_weights.data(), net.d_weights, net.total_weights_size * sizeof(float), cudaMemcpyDeviceToHost));
    // Can save h_weights now...

    // --- Cleanup ---
    free_network_memory(net);

    return 0;
}