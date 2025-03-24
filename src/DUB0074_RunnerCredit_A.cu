#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <cudaDefs.h>
#include <time.h>
#include <math.h>

using std::cout;
using std::endl;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int VECTOR_SIZE = 3;
constexpr unsigned int THREAD_COUNT = 256;

__constant__ __device__ float dMinFloat;
__constant__ __device__ float dMaxFloat;
//__constant__ __device__ int zeroVector[VECTOR_SIZE] = { 0,0,0 };//For now I consider only 3D vectors for simplicity -> Nebude vlastně třeba jelikož to je 0,0,0


__host__ float* createData(const unsigned int length)
{
	// Random number generator setup
	std::random_device rd;
	std::mt19937_64 mt(rd());
	// Trying to simulate unknown bounds
	std::uniform_real_distribution<float> lowerBound(0.1f, 1.0f);
	std::uniform_real_distribution<float> upperBound(2.0f, 3.0f);
	float lowerBoundValue = lowerBound(mt);
	float upperBoundValue = upperBound(mt);
	std::uniform_real_distribution<float> dist(lowerBoundValue, upperBoundValue);

	// Allocate memory for float array
	float* data;
	cudaMallocManaged(&data, length * sizeof(float));

	if (!data) {
		fprintf(stderr, "Memory allocation failed!\n");
		return nullptr;
	}

	// Populate the array with random float values
	for (unsigned int i = 0; i < length; i++) {
		data[i] = dist(mt);
	}

	// Ensure memory is synchronized if using unified memory
	cudaDeviceSynchronize();

	return data;
}
/*
Zjisti min a max hodnoty v matici
*/
__global__ void kernelFindMinMax(const float* __restrict__ matrix, const size_t m_size, const size_t n_size, float* __restrict__ minFloat, float* __restrict__ maxFloat)
{
	__shared__ float sMin[THREAD_COUNT];
	__shared__ float sMax[THREAD_COUNT];
	//unsigned int offset_x = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int offset_y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < m_size* n_size) // Here I check boundaries of matrix
	{
		sMin[threadIdx.x] = matrix[offset];
		sMax[threadIdx.x] = matrix[offset];
	}
	else
	{
		sMin[threadIdx.x] = FLT_MAX;
		sMax[threadIdx.x] = FLT_MIN;
	}
	__syncthreads();
	for (int i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if (threadIdx.x < i)
		{
			sMin[threadIdx.x] = fminf(sMin[threadIdx.x], sMin[threadIdx.x + i]);
			sMax[threadIdx.x] = fmaxf(sMax[threadIdx.x], sMax[threadIdx.x + i]);
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		if (sMin[0] < *minFloat)
			*minFloat = sMin[0];
		if (sMax[0] > *maxFloat)
			*maxFloat = sMax[0];
	}
}
__global__ void kernelDiscretize(const float* __restrict__ matrix, const size_t m_size, const size_t n_size, uint8_t* __restrict__ dMatrix)
{
	//int offset_x = threadIdx.x + blockIdx.x * blockDim.x;
	//int offset_y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset < m_size * n_size) // Here I check boundaries of matrix
	{
		float value = matrix[offset];
		float normalized_value = (value - dMinFloat) / (dMaxFloat - dMinFloat);
		dMatrix[offset] = static_cast<uint8_t>(normalized_value * 255); // Discretize to uint8_t
	}
}
// One thread = one vector
__global__ void kernelGetDistances(const uint8_t* __restrict__ matrix, const size_t m_size, const size_t n_size, float* origin, float* __restrict__ distances)
{
	unsigned int offset_x = threadIdx.x + blockIdx.x * blockDim.x;
	if (offset_x < m_size) // Here I check boundaries of matrix
	{
		distances[offset_x] = 0.0f;

		// Compute squared Euclidean distance for the vector
		for (unsigned int i = 0; i < n_size; i++) {
			uint8_t value = matrix[offset_x + i];
			uint8_t originValue = origin[i];
			distances[offset_x] += (value - originValue) * (value - originValue);
		}
	}
}

__global__ void kernelGetFarthest(float* __restrict__ distances, const size_t m_size, const size_t n_size, int* __restrict__ vectorIndex)
{
	unsigned int offset = threadIdx.x + blockIdx.x * blockDim.x;

	// Ensure we're within bounds
	if (offset < m_size)
	{
		// Get the current value of the farthest index (from vectorIndex[0])
		int currentIndex = vectorIndex[0];

		// Use atomicCAS to safely update the vectorIndex if the current distance is greater than the one at the current index
		// We only want to update vectorIndex[0] if distances[offset] > distances[currentIndex]
		while (distances[offset] > distances[currentIndex])
		{
			// Perform atomicCAS to check if vectorIndex[0] matches the currentIndex
			// If it does, we update it with the current offset; otherwise, we continue trying.
			currentIndex = atomicCAS(&vectorIndex[0], currentIndex, offset);

			// If currentIndex didn't match, it means another thread updated vectorIndex[0], so we need to re-check the condition
			if (currentIndex == vectorIndex[0]) {
				break;  // Exit the loop if no update was made
			}
		}
	}
}

int main(int argc, char* argv[])
{
	//Inicializace a alokace promennych

	initializeCUDA(deviceProp);
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	unsigned int m_size = 2 << 20;
	unsigned int n_size = VECTOR_SIZE;
	unsigned int threadCount = THREAD_COUNT;
	unsigned int blockCount = (m_size * n_size + threadCount - 1) / threadCount;
	uint8_t* dMatrix = nullptr;
	checkCudaErrors(cudaMallocManaged(&dMatrix, m_size * n_size * sizeof(uint8_t)));
	float* minFloat = nullptr;
	float* maxFloat = nullptr;
	checkCudaErrors(cudaMallocManaged(&minFloat, sizeof(float)));
	checkCudaErrors(cudaMallocManaged(&maxFloat, sizeof(float)));

	// Nastaveni default hodnot
	*minFloat = FLT_MAX;
	*maxFloat = FLT_MIN;
	float* hMatrix = createData(m_size * n_size);
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	kernelFindMinMax << <blockCount, threadCount >> > (hMatrix, m_size, n_size, minFloat, maxFloat);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Finding MinMax Float: %f ms\n", elapsedTime);
	cudaEventRecord(startEvent, 0);
	cudaEventSynchronize(startEvent);
	//Ted zname min a max a muzeme je nahrat do constant memory pro pouziti k diskretizaci
	checkCudaErrors(cudaMemcpyToSymbol(dMinFloat, minFloat, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(dMaxFloat, maxFloat, sizeof(float)));
	printf("Min: %f, Max: %f\n", *minFloat, *maxFloat);
	
	kernelDiscretize << <blockCount, threadCount >> > (hMatrix, m_size, n_size, dMatrix);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Discretization: %f ms\n", elapsedTime);
	cudaEventRecord(startEvent, 0);
	cudaEventSynchronize(startEvent);
	float* dDistances = nullptr;
	checkCudaErrors(cudaMallocManaged(&dDistances, sizeof(float) * m_size));
	float* origin = nullptr;
	checkCudaErrors(cudaMallocManaged(&origin, sizeof(float) * n_size));
	for (int i = 0; i < n_size; i++)
	{
		origin[i] = 0.0f;
	}
	kernelGetDistances << <(m_size + threadCount - 1) / threadCount, threadCount >> > (dMatrix, m_size, n_size, origin, dDistances); // One thread = one vector
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Getting distances from zero vector: %f ms\n", elapsedTime);
	cudaEventRecord(startEvent, 0);
	cudaEventSynchronize(startEvent);

	/*for (int i = 0; i < 10; i++)
	{
		printf("Distance %d: %f\n", i, dDistances[i]);
	}*/

	int* vectorIndex = nullptr;
	checkCudaErrors(cudaMallocManaged(&vectorIndex, sizeof(int)));
	*vectorIndex = 0;
	kernelGetFarthest << <(m_size + threadCount - 1) / threadCount, threadCount >> > (dDistances, m_size, n_size, vectorIndex);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Finding farthest vector: %f ms\n", elapsedTime);

	printf("Farthest vector index: %d\n", *vectorIndex);
	printf("Farthest vector distance: %f\n", dDistances[*vectorIndex]);
	printf("Farthest vector: ");
	for (int i = 0; i < n_size; i++)
	{
		printf("%d ", dMatrix[n_size * *vectorIndex + i]);
	}
	printf("\n");
	for (int i = 0; i < n_size; i++)
	{
		origin[i] = dMatrix[n_size * *vectorIndex + i]; //Set furthest vector as origin
	}
	kernelGetDistances << <(m_size + threadCount - 1) / threadCount, threadCount >> > (dMatrix, m_size, n_size, origin, dDistances); // One thread = one vector
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Getting distances from farthest vector: %f ms\n", elapsedTime);

	*vectorIndex = 0;
	kernelGetFarthest << <(m_size + threadCount - 1) / threadCount, threadCount >> > (dDistances, m_size, n_size, vectorIndex);
	checkCudaErrors(cudaDeviceSynchronize());  // Wait for kernel to finish
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Finding farthest vector: %f ms\n", elapsedTime);

	printf("Farthest vector distance: %f\n", dDistances[*vectorIndex]);
	printf("Farthest vector: ");
	for (int i = 0; i < n_size; i++)
	{
		printf("%d ", dMatrix[n_size * *vectorIndex + i]);
	}
	printf("\n");
	/*if (hMatrix)
		free(hMatrix);*/

	checkCudaErrors(cudaFree(minFloat));
	checkCudaErrors(cudaFree(maxFloat));
	checkCudaErrors(cudaFree(dMatrix));
	checkCudaErrors(cudaFree(dDistances));
	checkCudaErrors(cudaFree(vectorIndex));
	checkCudaErrors(cudaFree(origin));
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	
}
