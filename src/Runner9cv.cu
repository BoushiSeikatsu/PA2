#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <benchmark.h>

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int N = 1 << 22;
constexpr unsigned int MEMSIZE = N * sizeof(unsigned int);
constexpr unsigned int NO_LOOPS = 100;
constexpr unsigned int TPB = 256;
constexpr unsigned int GRID_SIZE = (N + TPB - 1) / TPB;

constexpr unsigned int NO_TEST_PHASES = 10;

void fillData(unsigned int *data, const unsigned int length)
{
	for (unsigned int i=0; i<length; i++)
	{
		data[i]= 1;
	}
}

void printData(const unsigned int *data, const unsigned int length)
{
	if (data ==0) return;
	for (unsigned int i=0; i<length; i++)
	{
		printf("%u ", data[i]);
	}
}


__global__ void kernel(const unsigned int *a, const unsigned int *b, const unsigned int length, unsigned int *c)
{
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//TODO:  thread block loop
	if (tid < length)
	{
		c[tid] = a[tid] + b[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 1. - single stream, async calling </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test1()
{
	unsigned int *a, *b, *c;
	unsigned int *da, *db, *dc;

	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE,cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc( (void**)&da, MEMSIZE );
	cudaMalloc( (void**)&db, MEMSIZE );
	cudaMalloc( (void**)&dc, MEMSIZE );

	//TODO: create stream

	cudaStream_t stream;
	//cudaStream_t stream2;
	cudaStreamCreate(&stream);
	//cudaStreamCreate(&stream2);
	
	auto lambda = [&]()
	{	
		unsigned int dataOffset = 0;
		for (int i = 0; i < NO_LOOPS; i++)
		{
			cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
			cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
			kernel << <GRID_SIZE, TPB >> > (da, db, N, dc);
			cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);
			dataOffset += N;
			//TODO:  copy a->da, b->db
			//TODO:  run the kernel in the stream
			//TODO:  copy dc->c

		}
	};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream); // wait for stream to finish
	cudaStreamDestroy(stream);
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	//printData(c, 100);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 2. - two streams - depth first approach </summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test2()
{
	unsigned int* a, * b, * c;
	unsigned int* da, * db, * dc;
	unsigned int* a2, * b2, * c2;
	unsigned int* da2, * db2, * dc2;
	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	cudaHostAlloc((void**)&a2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a2, NO_LOOPS * N);
	fillData(b2, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da2, MEMSIZE);
	cudaMalloc((void**)&db2, MEMSIZE);
	cudaMalloc((void**)&dc2, MEMSIZE);

	//TODO: create stream

	cudaStream_t stream;
	cudaStream_t stream2;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream2);

	auto lambda = [&]()
		{
			unsigned int dataOffset = 0;
			unsigned int dataOffset2 = N * (NO_LOOPS- 1 );
			for (int i = 0; i < NO_LOOPS / 2 ; i++)
			{
				cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
				cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
				kernel << <GRID_SIZE, TPB >> > (da, db, N, dc);
				cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);

				cudaMemcpyAsync(da2, &a2[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2);
				cudaMemcpyAsync(db2, &b2[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2);
				kernel << <GRID_SIZE, TPB >> > (da2, db2, N, dc2);
				cudaMemcpyAsync(&c[dataOffset2], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream2);

				dataOffset += N;
				dataOffset2 -= N;
				//TODO:  copy a->da, b->db
				//TODO:  run the kernel in the stream
				//TODO:  copy dc->c

			}
		};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream); // wait for stream to finish
	cudaStreamDestroy(stream);
	cudaStreamSynchronize(stream2); // wait for stream to finish
	cudaStreamDestroy(stream2);
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	//printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	cudaFree(da2);
	cudaFree(db2);
	cudaFree(dc2);

	cudaFreeHost(a2);
	cudaFreeHost(b2);
	cudaFreeHost(c2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests 3. - two streams - breadth first approach</summary>
////////////////////////////////////////////////////////////////////////////////////////////////////
void test3()
{
	unsigned int* a, * b, * c;
	unsigned int* da, * db, * dc;
	unsigned int* a2, * b2, * c2;
	unsigned int* da2, * db2, * dc2;
	// paged-locked allocation
	cudaHostAlloc((void**)&a, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a, NO_LOOPS * N);
	fillData(b, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da, MEMSIZE);
	cudaMalloc((void**)&db, MEMSIZE);
	cudaMalloc((void**)&dc, MEMSIZE);

	cudaHostAlloc((void**)&a2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&b2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);
	cudaHostAlloc((void**)&c2, NO_LOOPS * MEMSIZE, cudaHostAllocDefault);

	fillData(a2, NO_LOOPS * N);
	fillData(b2, NO_LOOPS * N);

	// Data chunks on GPU
	cudaMalloc((void**)&da2, MEMSIZE);
	cudaMalloc((void**)&db2, MEMSIZE);
	cudaMalloc((void**)&dc2, MEMSIZE);

	//TODO: create stream

	cudaStream_t stream;
	cudaStream_t stream2;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream2);

	auto lambda = [&]()
		{
			unsigned int dataOffset = 0;
			unsigned int dataOffset2 = N * (NO_LOOPS - 1);
			for (int i = 0; i < NO_LOOPS; i += 2)
			{
				cudaMemcpyAsync(da, &a[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
				cudaMemcpyAsync(da2, &a2[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2);
				cudaMemcpyAsync(db, &b[dataOffset], MEMSIZE, cudaMemcpyHostToDevice, stream);
				cudaMemcpyAsync(db2, &b2[dataOffset2], MEMSIZE, cudaMemcpyHostToDevice, stream2);
				kernel << <GRID_SIZE, TPB >> > (da, db, N, dc);
				kernel << <GRID_SIZE, TPB >> > (da2, db2, N, dc2);
				cudaMemcpyAsync(&c[dataOffset], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream);
				cudaMemcpyAsync(&c[dataOffset2], dc, MEMSIZE, cudaMemcpyDeviceToHost, stream2);
				dataOffset += N;
				dataOffset2 -= N;
				//TODO:  copy a->da, b->db
				//TODO:  run the kernel in the stream
				//TODO:  copy dc->c

			}
		};
	float gpuTime = GPUTIME(NO_TEST_PHASES, lambda());

	cudaStreamSynchronize(stream); // wait for stream to finish
	cudaStreamDestroy(stream);
	cudaStreamSynchronize(stream2); // wait for stream to finish
	cudaStreamDestroy(stream2);
	cudaDeviceSynchronize();
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", __PRETTY_FUNCTION__, gpuTime);

	//printData(c, 100);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	cudaFree(da2);
	cudaFree(db2);
	cudaFree(dc2);

	cudaFreeHost(a2);
	cudaFreeHost(b2);
	cudaFreeHost(c2);
}


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	test1();
	test2();
	test3();

	return 0;
}
