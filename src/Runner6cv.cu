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
__constant__ int dfilter_x[3][3];
__constant__ int dfilter_y[3][3];
using DT = float;

__host__ TextureInfo createTextureObjectFrom2DArray(const ImageInfo<DT>& ii)
{
	TextureInfo ti;

	// Size info
	ti.size = { ii.width, ii.height, 1 };
	//Texture Data settings
	ti.texChannelDesc = cudaCreateChannelDesc<DT>();  // cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaMallocArray(&ti.texArrayData, &ti.texChannelDesc, ii.width, ii.height));
	checkCudaErrors(cudaMemcpyToArray(ti.texArrayData, 0, 0, ii.dPtr, ii.pitch * ii.height, cudaMemcpyDeviceToDevice));

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

template<bool normalizeTexel>__global__ void createNormalmap(const cudaTextureObject_t srcTex, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int dstPitchInElements, uchar3* dst)
{
	//TODO
	const unsigned int offset_x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int offset_y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int position = offset_y * dstPitchInElements + offset_x;
	int sum_x = 0;
	int sum_y = 0;
	if ((offset_x >= srcWidth) || (offset_y >= srcHeight)) return;
	if (offset_y - 1 > 0)
	{
		if (offset_x - 1 > 0)
		{
			sum_x += tex2D<float>(srcTex, offset_x - 1, offset_y - 1) * dfilter_x[0][0];
			sum_y += tex2D<float>(srcTex, offset_x - 1, offset_y - 1) * dfilter_y[0][0];

		}
		sum_x += tex2D<float>(srcTex, offset_x, offset_y - 1) * dfilter_x[0][1];
		sum_y += tex2D<float>(srcTex, offset_x, offset_y - 1) * dfilter_y[0][1];
		sum_x += tex2D<float>(srcTex, offset_x + 1, offset_y - 1) * dfilter_x[0][2];
		sum_y += tex2D<float>(srcTex, offset_x + 1, offset_y - 1) * dfilter_y[0][2];
	}
	if (offset_x - 1 > 0)
	{
		sum_x += tex2D<float>(srcTex, offset_x - 1, offset_y) * dfilter_x[1][0];
		sum_y += tex2D<float>(srcTex, offset_x - 1, offset_y) * dfilter_y[1][0];
		sum_x += tex2D<float>(srcTex, offset_x - 1, offset_y + 1) * dfilter_x[2][0];
		sum_y += tex2D<float>(srcTex, offset_x - 1, offset_y + 1) * dfilter_y[2][0];
	}
	sum_x += tex2D<float>(srcTex, offset_x, offset_y + 1) * dfilter_x[2][1];
	sum_y += tex2D<float>(srcTex, offset_x, offset_y + 1) * dfilter_y[2][1];
	sum_x += tex2D<float>(srcTex, offset_x + 1, offset_y) * dfilter_x[1][2];
	sum_y += tex2D<float>(srcTex, offset_x + 1, offset_y) * dfilter_y[1][2];
	sum_x += tex2D<float>(srcTex, offset_x + 1, offset_y + 1) * dfilter_x[2][2];
	sum_y += tex2D<float>(srcTex, offset_x + 1, offset_y + 1) * dfilter_y[2][2];
	float3 d;
	d.x = sum_x;
	d.y = sum_y;
	d.z = 1.f/2.f;
	if (normalizeTexel)
	{
		d = normalize(d);
	}
	d.x = (d.x + 1) * 127.5;
	d.y = (d.y + 1) * 127.5;
	d.z = (d.z + 1) * 127.5;
	dst[position] = make_uchar3(static_cast<uint8_t>(d.z), static_cast<uint8_t>(d.y), static_cast<uint8_t>(d.x));
}

void saveTexImage(const char* imageFileName, const uint32_t dstWidth, const uint32_t dstHeight, const uint32_t dstPitch, const uchar3* dstData)
{
	FIBITMAP* tmp = FreeImage_Allocate(dstWidth, dstHeight, 24);
	unsigned int tmpPitch = FreeImage_GetPitch(tmp);					// FREEIMAGE align row data ... You have to use pitch instead of width
	checkCudaErrors(cudaMemcpy2D(FreeImage_GetBits(tmp), tmpPitch, dstData, dstPitch, dstWidth * 3, dstHeight, cudaMemcpyDeviceToHost));
	//FreeImage_Save(FIF_BMP, tmp, imageFileName, 0);
	ImageManager::GenericWriter(tmp, imageFileName, FIF_BMP);
	FreeImage_Unload(tmp);
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
	int filter_x[3][3] = { { -1, 0, 1 },{ -2, 0, 2 },{ -1, 0, 1 } };
	int filter_y[3][3] = { { -1, -2, -1 },{ 0, 0, 0 },{ 1, 2, 1 } };
	cudaMemcpyToSymbol(dfilter_x, filter_x, 3 * 3 * sizeof(int));
	cudaMemcpyToSymbol(dfilter_y, filter_y, 3 * 3 * sizeof(int));
	// STEP 1 - load raw image data, HOST->DEVICE, with/without pitch
	ImageInfo<DT> src;
	prepareData<false>("C:\\Users\\dub0074\\Documents\\PA2\\src/terrain3Kx3K.tif", src);

	// STEP 2 - create texture from the raw data
	TextureInfo tiSrc = createTextureObjectFrom2DArray(src);
	// SETP 3 - allocate pitch memory to store output image data
	
	size_t dstPitch;
	uchar3* dst = 0;
	cudaMallocPitch((void**)&dst, &dstPitch, src.width * sizeof(uchar3), src.height);
	// STEP 4 - create normal map
	dim3 block = { TPB_1D, TPB_1D,1 };
	dim3 grid{ (src.width + TPB_1D - 1) / TPB_1D, (src.height + TPB_1D - 1) / TPB_1D,  1 };
	createNormalmap<true> << <grid, block >> > (tiSrc.texObj, src.width, src.height, dstPitch / sizeof(uchar3), dst);
	// STEP 5 - save the normal map
	saveTexImage("C:\\Users\\dub0074\\Documents\\PA2\\src/normalmap.bmp", src.width, src.height, dstPitch, dst);
	// SETP 6 - release unused data
	if (tiSrc.texObj) checkCudaErrors(cudaDestroyTextureObject(tiSrc.texObj));
	if (tiSrc.texArrayData) checkCudaErrors(cudaFreeArray(tiSrc.texArrayData));
	//if (src.) cudaFree(src.dPtr);
	if (dst) cudaFree(dst);
	cudaDeviceSynchronize();
	error = cudaGetLastError();

	FreeImage_DeInitialise();
}
