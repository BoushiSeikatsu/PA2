#include <cudaDefs.h>
#include <time.h>
#include <math.h>
#include <random>

//WARNING!!! Do not change TPB and NO_FORCES for this demo !!!
constexpr unsigned int TPB = 128;
constexpr unsigned int NO_FORCES = 256;
constexpr unsigned int NO_RAIN_DROPS = 1 << 20;

constexpr unsigned int MEM_BLOCKS_PER_THREAD_BLOCK = 8;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace std;

__host__ float3 *createData(const unsigned int length)
{
	//TODO: Generate float3 vectors. You can use 'make_float3' method.
	random_device rd;
	mt19937_64 mt(rd());
	uniform_real_distribution<float> dist(0.0f, 1.0f);
	float3* data = static_cast<float*>(::new_handler(length * sizeof(float3)));
	float3* ptr;
	return data;
}

__host__ void printData(const float3 *data, const unsigned int length)
{
	if (data == 0) return;
	const float3 *ptr = data;
	for (unsigned int i = 0; i<length; i++, ptr++)
	{
		printf("%5.2f %5.2f %5.2f ", ptr->x, ptr->y, ptr->z);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Sums the forces to get the final one using parallel reduction. 
/// 		    WARNING!!! The method was written to meet input requirements of our example, i.e. 128 threads and 256 forces  </summary>
/// <param name="dForces">	  	The forces. </param>
/// <param name="noForces">   	The number of forces. </param>
/// <param name="dFinalForce">	[in,out] If non-null, the final force. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void reduce(const float3 * __restrict__ dForces, const unsigned int noForces, float3* __restrict__ dFinalForce)
{
	__shared__ float3 sForces[TPB];					//SEE THE WARNING MESSAGE !!!
	unsigned int tid = threadIdx.x;
	unsigned int next = TPB;						//SEE THE WARNING MESSAGE !!!

	float3* src = &sForces[tid];
	*src = dForces[tid];
	float3* src2 = (float3*)&dForces[tid + next];
	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;
	__syncthreads();
	next >>= 1;
	if (tid >= next) return;
	src2 = src + next;
	src->x += src2->x;
	src->y += src2->y;
	src->z += src2->z;
	__syncthreads();
	next >>= 1;
	if (tid >= next) return;
	volatile float3* vscr = &sForces[tid];
	volatile float3* vscr2 = vscr + next;
	vscr->x += vscr2->x;
	vscr->y += vscr2->y;
	vscr->z += vscr2->z;

	//TODO: Make the reduction
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Adds the FinalForce to every Rain drops position. </summary>
/// <param name="dFinalForce">	The final force. </param>
/// <param name="noRainDrops">	The number of rain drops. </param>
/// <param name="dRainDrops"> 	[in,out] If non-null, the rain drops positions. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void add(const float3* __restrict__ dFinalForce, const unsigned int noRainDrops, float3* __restrict__ dRainDrops)
{
	//TODO: Add the FinalForce to every Rain drops position.
}


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);

	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	float3 *hForces = createData(NO_FORCES);
	float3 *hDrops = createData(NO_RAIN_DROPS);

	float3 *dForces = nullptr;
	float3 *dDrops = nullptr;
	float3 *dFinalForce = nullptr;

	checkCudaErrors(cudaMalloc((void**)&dForces, NO_FORCES * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dForces, hForces, NO_FORCES * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dDrops, NO_RAIN_DROPS * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(dDrops, hDrops, NO_RAIN_DROPS * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&dFinalForce, sizeof(float3)));

	KernelSetting ksReduce;

	//TODO: ... Set ksReduce
	
	KernelSetting ksAdd;
	//TODO: ... Set ksAdd
	
	for (unsigned int i = 0; i<1000; i++)
	{
		reduce<<<ksReduce.dimGrid, ksReduce.dimBlock>>>(dForces, NO_FORCES, dFinalForce);
		add<<<ksAdd.dimGrid, ksAdd.dimBlock>>>(dFinalForce, NO_RAIN_DROPS, dDrops);
	}

	checkDeviceMatrix<float>((float*)dFinalForce, sizeof(float3), 1, 3, "%5.2f ", "Final force");
	// checkDeviceMatrix<float>((float*)dDrops, sizeof(float3), NO_RAIN_DROPS, 3, "%5.2f ", "Final Rain Drops");

	if (hForces)
		free(hForces);
	if (hDrops)
		free(hDrops);

	checkCudaErrors(cudaFree(dForces));
	checkCudaErrors(cudaFree(dDrops));

	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);

	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	printf("Time to get device properties: %f ms", elapsedTime);
}
