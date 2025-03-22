// includes, cudaimageWidth
#include <cudaDefs.h>

#include <helper_math.h>			// normalize method

#include <imageManager.h>
#include <imageUtils.cuh>
#include <benchmark.h>

#define TPB_1D 8						// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D			// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

using namespace gpubenchmark;
using DT = float;


__host__ TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT>& ii)
{
	TextureInfo ti;

	// Size info
	ti.size = { ii.width, ii.height, 1 };
	//Texture Data settings
	ti.texChannelDesc = cudaCreateChannelDesc<DT>();  // cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(&ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height));
	checkCudaErrors(cudaMemcpyToArray(ti.texArrayData, 0,0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice));

	// Specify texture resource
	ti.resDesc.resType = cudaResourceTypeArray;
	ti.resDesc.res.array.array = ti.texArrayData;

	// Specify texture object parameters
	ti.texDesc.addressMode[0] = cudaAddressModeClamp;
	ti.texDesc.addressMode[1] = cudaAddressModeClamp;
	ti.texDesc.filterMode = cudaFilterModePoint;
	ti.texDesc.readMode = cudaReadModeElementType;
	ti.texDesc.normalizedCoords = false;

	// Create texture object
	checkCudaErrors(cudaCreateTextureObject(&ti.texObj, &ti.resDesc, &ti.texDesc, NULL));

	return ti;
}

__global__ void texKernel(const cudaTextureObject_t srcTex, const unsigned int srcWidth, const unsigned int srcHeight, float* dst)
{
	int offset_x = threadIdx.x + blockIdx.x * blockDim.x;
	int offset_y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((offset_x >= srcWidth) || (offset_y >= srcHeight)) return;
	dst[srcWidth * offset_y + offset_x] = tex2D<float>(srcTex, offset_x, offset_y);
}


int main(int argc, char *argv[])
{
	initializeCUDA(deviceProp);
	FreeImage_Initialise();

	// STEP 1 - load raw image data, HOST->DEVICE, with/without pitch
	ImageInfo<DT> src;
	prepareData<false>("C:\\Users\\dub0074\\Documents\\PA2\\src\\terrain10x10.tif", src);

	// STEP 2 - create texture from the raw data
	TextureInfo tiSrc = createTextureObjectFrom2DArray(src);

	// STEP 3 - DO SOMETHING WITH THE TEXTURE
	dim3 block = { TPB_1D, TPB_1D,1 };
	dim3 grid{ (src.width + TPB_1D - 1) / TPB_1D, (src.height + TPB_1D - 1) / TPB_1D,  1 };
	float* dst = nullptr;
	cudaMalloc((void**)&dst, src.width * src.height * sizeof(float));
	float gpuTime = GPUTIME(1, texKernel <<<grid, block>>> (tiSrc.texObj, src.width, src.height, dst));
	printf("\x1B[93m[GPU time] %s: %f ms\033[0m\n", "getBest", gpuTime);
	checkDeviceMatrix<float>(dst, src.width * sizeof(float), src.height, src.width, "%6.1f ", "dst");

	// SETP 4 - release unused data
	if (tiSrc.texObj) checkCudaErrors(cudaDestroyTextureObject(tiSrc.texObj));
	if (tiSrc.texArrayData) checkCudaErrors(cudaFreeArray(tiSrc.texArrayData));
	if (src.dPtr) cudaFree(src.dPtr);
	if (dst) cudaFree(dst);

	cudaDeviceSynchronize();
	error = cudaGetLastError();

	FreeImage_DeInitialise();
}
