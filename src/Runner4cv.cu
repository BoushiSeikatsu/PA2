#include <cudaDefs.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <random>
using namespace std;



using namespace std;
const int patternLength = 16;
__host__ float* createData(const unsigned int length)
{
	// Random number generator setup
	std::random_device rd;
	std::mt19937_64 mt(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

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

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

#pragma region CustomStructure
typedef struct __align__(8) CustomStructure
{
public:
	int dim;				//dimension
	int noRecords;			//number of Records

	CustomStructure& operator=(const CustomStructure& other)
	{
		dim = other.dim;
		noRecords = other.noRecords;
		return *this;
	}

	inline void print()
	{
		printf("Dimension: %u\n", dim);
		printf("Number of Records: %u\n", noRecords);
	}
}CustomStructure;
#pragma endregion

__constant__ __device__ int dScalarValue;
__constant__ __device__ struct CustomStructure dCustomStructure;
__constant__ __device__ int dConstantArray[20];
__constant__ __device__ float dPattern[patternLength];

__global__ void kernelConstantStruct(int *data, const unsigned int dataLength)
{
	unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadOffset < dataLength)
		data[threadOffset] = dCustomStructure.dim;
}

__global__ void kernelConstantArray(int *data, const unsigned int dataLength)
{
	unsigned int threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadOffset < dataLength)
		data[threadOffset] = dConstantArray[0];
}

__global__ void kernelFindPattern(const float* __restrict__ reference, const size_t referenceLength, const size_t patternLength, bool* __restrict__ output)
{
	const unsigned int offset = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int blockOffset = (2e23 + 16 + 1)/
	if (offset + patternLength <= referenceLength)
	{
		bool isMatching = true;
		for (int i = 0; i < patternLength; i++)
		{
			isMatching &= (dPattern[i] != reference[offset + i]); //Takhle se nam nerozbije warp (op na optimalizaci) 
			//if (dPattern[i] != reference[offset + i])
			//{
				//isMatching = false;
				//Kdybych tady dal break tak bych si rozbil warp, už ale ten warp rozbijím tím ifem, jde o to že nejedou najednou pararelně
			//}
		}
		output[offset] = isMatching;
	}
	else
	{
		output[offset] = false;
	}
}
//(l+m+1) / m -> l je velikost plochy co chci pokrýt, m je velikost kterou pokrýváme
int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	float* pattern = createData(16);
	cudaMemcpyToSymbol(dPattern, pattern, sizeof(float)*patternLength);
	float tmp[patternLength];
	cudaMemcpyFromSymbol(tmp, dPattern, sizeof(float) * patternLength);
	for(const auto& i : tmp)
	{
		std::cout << i << " ";
	}
	std::cout << std::endl;
	const int referenceLength = 2e23;
	float* dReference = nullptr;
	cudaMalloc((void**)(&dReference), sizeof(float) * referenceLength);
	float* hReference = createData(referenceLength);
	const int threadNumber = 256;
	const int numBlocks = (referenceLength + threadNumber + 1) / threadNumber;
	bool* output = nullptr;
	cudaMalloc((void**)(&dReference), sizeof(bool) * referenceLength);
	kernelFindPattern << <numBlocks, threadNumber >> > (hReference, referenceLength, patternLength, output);
	for (int i = 0; i < referenceLength; i++)
	{
		if (output[i])
		{
			cout << i << endl;
			break;
		}
	}

	/*
	int scalarValue = 10;
	cudaMemcpyToSymbol(dScalarValue, &scalarValue, sizeof(int));
	int newValue;
	cudaMemcpyFromSymbol(&newValue, dScalarValue, sizeof(int));
	printf("Copied value: %d", newValue);

	CustomStructure tmp = CustomStructure();
	tmp.dim = 2;
	tmp.noRecords = 5;
	checkCudaErrors(cudaMemcpyToSymbol(dCustomStructure, &tmp,sizeof(tmp)));
	CustomStructure result = CustomStructure();
	checkCudaErrors(cudaMemcpyFromSymbol(&result, dCustomStructure, sizeof(dCustomStructure)));
	result.print();

	int tmpArray[10] = {0,1,2,3,4,5,6,7,8,9};
	cudaMemcpyToSymbol(dConstantArray, tmpArray, sizeof(int) * 10);
	int resultArray[10] = {0};
	cudaMemcpyFromSymbol(resultArray, dConstantArray, sizeof(int) * 10);
	for (const auto& i : resultArray)
	{
		std::cout << i;
	}
	std::cout << std::endl;
	//Test 0 - scalar Value
	//Test 1 - structure
	//Test2 - array*/

}

/*
CREDIT TASK -> JE TŘEBA NASTAVIT LIMITY, POKUD CHCI VYUŽÍT CONSTANT PAMĚT, TAK BUDU MÍT MALÉ PAMĚTI MALÉ PROMĚNNÉ, POKUD VŠECKO BUDE V GLOBAL MEMORY TAK ZASE VELKÉ, JE TO JENOM O TOM SI TO OBHÁJIT. 
ZÁLEŽÍ NA TOM JAK PRACUJEME S PAMĚTMI A JAK K NIM PŘISTUPUJEME
POKUD BYCH BYL UPLNĚ CLUELESS TAK 19 SE DÁ ZKONZULTOVAT 

INFO O PROJEKTECH JEŠTĚ
SOLO NEBO DUO
DO PŘIŠTĚ SI ROZMYSLET CO BUDU CHTÍT DĚLAT A JAK TO BUDU MÍT DATOVĚ ROZDĚLENÉ, UKÁZAT NA PAPÍŘE JAK TY DTAA V PAMĚTI BUDOU VYPADAT, ZKONZULTOVAT AŤ NEJDEME SLEPOU VĚTVÍ 
CO BYCH CHTĚL DĚLAT? BUDU DĚLAT TO A TO, TAKOVOU MÁM PŘEDSTAVU O DATOVÉ SESTAVĚ, ON PAK DÁ FEEDBACK JESTLI TO JE STUPID NEBO NE

CHCE TO KONZULTOVAT VŽDYCKY NA CVIKU, HODNĚ TO ULEHČÍ PROJEKT, PROTOŽE SE TO VYBRAINSTORMI NA CVIKU 
*/
