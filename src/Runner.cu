#include <glew.h>
#include <freeglut.h>
#include <FreeImage.h>

#include <cudaDefs.h> 

#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h> 
#include <vector_types.h> 

#include <imageManager.h>
#include <benchmark.h>

#include <iostream> 
#include <string>   
#include <vector>   
#include <cmath>    

// Define M_PI if not available (helper_math.h might provide CUDART_PI_F)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TPB_1D 8									// ThreadsPerBlock in one dimension
#define TPB_2D TPB_1D*TPB_1D						// ThreadsPerBlock = TPB_1D*TPB_1D (2D block)

// File names
const char* SOURCE_IMG_FN = "C:\\Users\\BoushiPC\\Documents\\PythonScripts\\PA2\\src/CreditTaskD_src.png";
const char* PATTERN0_IMG_FN = "C:\\Users\\BoushiPC\\Documents\\PythonScripts\\PA2\\src/CreditTaskD_pattern0.png";
const char* PATTERN1_IMG_FN = "C:\\Users\\BoushiPC\\Documents\\PythonScripts\\PA2\\src/CreditTaskD_pattern1.png";


cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

bool matchExecutionDone = false; // Flag to run matching only once

using DT = uchar4;

// Struct to store match results
struct FoundMatchInfo {
	int x;
	int y;
	int rotation_idx; // 0:0, 1:90, 2:180, 3:270 deg clockwise
};


//OpenGL Data
struct GLData
{
	unsigned int imageWidth;    // Source image width
	unsigned int imageHeight;   // Source image height
	unsigned int imageBPP;
	unsigned int imagePitch;

	unsigned int pboID;
	unsigned int textureID;     // For source image display
	unsigned int viewportWidth = 1024;
	unsigned int viewportHeight = 1024;
};
GLData gl;


struct CudaSrcImageData
{
	cudaTextureDesc			texDesc;
	cudaArray_t				texArrayData;       // Mapped from GL texture
	cudaResourceDesc		resDesc;
	cudaChannelFormatDesc	texChannelDesc;
	cudaTextureObject_t		texObj;             // Source image texture object

	cudaGraphicsResource_t  texResource;        // Interop for GL Texture (source image)
	cudaGraphicsResource_t	pboResource;

	CudaSrcImageData()
	{
		memset(this, 0, sizeof(CudaSrcImageData));
	}
};
CudaSrcImageData cd_src; 

// Struct to manage CUDA resources for pattern images (loaded directly, no GL interop)
struct CudaPatternData {
	cudaTextureObject_t texObj = 0;
	cudaArray_t         texArray = nullptr;
	unsigned int        width = 0;
	unsigned int        height = 0;

	void loadPattern(const char* filename) {
		printf("Loading pattern: %s\n", filename);
		FIBITMAP* h_img = ImageManager::GenericLoader(filename, 0);
		if (!h_img) {
			fprintf(stderr, "ERROR: Could not load pattern image %s\n", filename);
			exit(EXIT_FAILURE);
		}

		unsigned int bpp = FreeImage_GetBPP(h_img);
		if (bpp != 32) {
			printf("Converting pattern %s to 32bpp...\n", filename);
			FIBITMAP* temp = FreeImage_ConvertTo32Bits(h_img);
			FreeImage_Unload(h_img);
			if (!temp) {
				fprintf(stderr, "ERROR: FreeImage_ConvertTo32Bits failed for pattern %s!\n", filename);
				exit(EXIT_FAILURE);
			}
			h_img = temp;
		}

		width = FreeImage_GetWidth(h_img);
		height = FreeImage_GetHeight(h_img);
		BYTE* h_pixelData = FreeImage_GetBits(h_img);
		unsigned int pitch = FreeImage_GetPitch(h_img);

		if (!h_pixelData) {
			fprintf(stderr, "ERROR: FreeImage_GetBits returned NULL for pattern %s!\n", filename);
			FreeImage_Unload(h_img);
			exit(EXIT_FAILURE);
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
		checkCudaErrors(cudaMallocArray(&texArray, &channelDesc, width, height));
		// Use pitch from FreeImage for cudaMemcpy2DToArray
		checkCudaErrors(cudaMemcpy2DToArray(texArray, 0, 0, h_pixelData, pitch,
			width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

		FreeImage_Unload(h_img); // Free host bitmap memory

		cudaResourceDesc resDesc_pattern;
		memset(&resDesc_pattern, 0, sizeof(resDesc_pattern));
		resDesc_pattern.resType = cudaResourceTypeArray;
		resDesc_pattern.res.array.array = texArray;

		cudaTextureDesc texDesc_pattern;
		memset(&texDesc_pattern, 0, sizeof(texDesc_pattern));
		texDesc_pattern.addressMode[0] = cudaAddressModeClamp; // Or cudaAddressModeBorder for out-of-bounds access
		texDesc_pattern.addressMode[1] = cudaAddressModeClamp;
		texDesc_pattern.filterMode = cudaFilterModePoint;    // Exact pixel matches
		texDesc_pattern.readMode = cudaReadModeElementType;  // Read as uchar4
		texDesc_pattern.normalizedCoords = 0;                // Use integer coordinates

		checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc_pattern, &texDesc_pattern, NULL));
		printf("Pattern %s loaded into CUDA texture: %u x %u\n", filename, width, height);
	}

	void release() {
		if (texObj) cudaDestroyTextureObject(texObj);
		texObj = 0;
		if (texArray) cudaFreeArray(texArray);
		texArray = nullptr;
		width = 0;
		height = 0;
	}
};
CudaPatternData pattern0_data;
CudaPatternData pattern1_data;

// GPU pointers for match results
FoundMatchInfo* d_foundMatchPattern0 = nullptr;
FoundMatchInfo* d_foundMatchPattern1 = nullptr;


#pragma region CUDA Routines

// Helper function to compare uchar4 colors (ignoring alpha, assuming .x=R, .y=G, .z=B)
__device__ inline bool colorsMatch(const uchar4 c1, const uchar4 c2) {
	return c1.x == c2.x && c1.y == c2.y && c1.z == c2.z;
}

// CUDA Kernel for pattern matching with rotation
__global__ void patternMatchKernel(
	cudaTextureObject_t srcTex, int srcW, int srcH,
	cudaTextureObject_t patTex, int patW, int patH,
	FoundMatchInfo* d_matchInfo // Output: {x, y, rotation_idx}
) {
	unsigned int s_x = blockIdx.x * blockDim.x + threadIdx.x; // Candidate top-left x in source
	unsigned int s_y = blockIdx.y * blockDim.y + threadIdx.y; // Candidate top-left y in source

	// Early exit if a match has already been found by another thread block / warp
	// Note: d_matchInfo->x is read. This read should be volatile or use atomicRead for strong guarantees,
	// but for performance, a dirty read is often acceptable in such "optimistic" termination.
	if (d_matchInfo->x != -1) {
		return;
	}

	// Iterate through 4 rotations: 0, 90, 180, 270 degrees clockwise
	for (int rot_idx = 0; rot_idx < 4; ++rot_idx) {
		int current_match_W, current_match_H; // Effective dimensions of pattern after rotation

		if (rot_idx == 0 || rot_idx == 2) { // 0 or 180 degrees
			current_match_W = patW;
			current_match_H = patH;
		}
		else { // 90 or 270 degrees
			current_match_W = patH; // Width becomes height
			current_match_H = patW; // Height becomes width
		}

		// Check if the rotated pattern, anchored at (s_x, s_y), fits within source image boundaries
		if (s_x + current_match_W > srcW || s_y + current_match_H > srcH) {
			continue; // This rotation at this (s_x, s_y) goes out of bounds
		}

		bool all_pixels_match_this_rotation = true;
		// Iterate over each pixel of the original (unrotated) pattern
		for (int p_y = 0; p_y < patH; ++p_y) {
			for (int p_x = 0; p_x < patW; ++p_x) {
				uchar4 patPixel = tex2D<uchar4>(patTex, p_x, p_y);

				// Handle transparency in pattern: if pattern pixel is (mostly) transparent, skip comparison
				if (patPixel.w < 32) { // Alpha threshold (0-255). Low value means mostly transparent.
					continue;
				}

				int src_read_x, src_read_y; // Coordinates in source image to compare against

				// Map pattern pixel (p_x, p_y) to corresponding coordinate in the source
				// based on (s_x, s_y) and current rotation
				switch (rot_idx) {
				case 0: // 0 degrees
					src_read_x = s_x + p_x;
					src_read_y = s_y + p_y;
					break;
				case 1: // 90 degrees clockwise: (p_x, p_y) -> (p_y, patW - 1 - p_x) relative to new frame
					src_read_x = s_x + p_y;
					src_read_y = s_y + (patW - 1 - p_x);
					break;
				case 2: // 180 degrees: (p_x, p_y) -> (patW - 1 - p_x, patH - 1 - p_y)
					src_read_x = s_x + (patW - 1 - p_x);
					src_read_y = s_y + (patH - 1 - p_y);
					break;
				case 3: // 270 degrees clockwise: (p_x, p_y) -> (patH - 1 - p_y, p_x)
					src_read_x = s_x + (patH - 1 - p_y);
					src_read_y = s_y + p_x;
					break;
				default: // Should not happen
					all_pixels_match_this_rotation = false; // Safety
					break;
				}
				if (!all_pixels_match_this_rotation) break;


				// Boundary check for read coordinates (should be covered by earlier check, but good for safety)
				// This check is technically redundant if the (s_x + current_match_W > srcW) logic is correct
				if (src_read_x < 0 || src_read_x >= srcW || src_read_y < 0 || src_read_y >= srcH) {
					all_pixels_match_this_rotation = false;
					break;
				}

				uchar4 srcPixel = tex2D<uchar4>(srcTex, src_read_x, src_read_y);

				if (!colorsMatch(patPixel, srcPixel)) {
					all_pixels_match_this_rotation = false;
					break; // Mismatch for this pattern pixel, try next rotation or (s_x, s_y)
				}
			}
			if (!all_pixels_match_this_rotation) {
				break; // Mismatch in this row of pattern, try next rotation
			}
		}

		if (all_pixels_match_this_rotation) {
			// Match found! Atomically update the global result structure.
			// atomicCAS returns the OLD value at d_matchInfo->x.
			// If old value was -1, this thread is the first to find a match.
			int old_x = atomicCAS(&(d_matchInfo->x), -1, s_x);
			if (old_x == -1) {
				// This thread successfully claimed the match. Now set y and rotation_idx.
				// No atomics needed for y and rotation_idx if x acts as the lock.
				atomicExch(&(d_matchInfo->y), s_y);
				atomicExch(&(d_matchInfo->rotation_idx), rot_idx);
			}
			// Even if this thread wasn't the first, a match is found for this (s_x,s_y)
			// The task requires one position, so this thread's work for this (s_x,s_y) is done.
			// And since a global match is found (either by this thread or another), it can return.
			return;
		}
	}
}


void cudaPatternMatchWorker()
{
	if (matchExecutionDone) return;

	printf("Starting pattern matching...\n");


	// Initialize result structs on GPU to {-1, -1, -1}
	FoundMatchInfo h_initialMatch = { -1, -1, -1 };
	checkCudaErrors(cudaMemcpy(d_foundMatchPattern0, &h_initialMatch, sizeof(FoundMatchInfo), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_foundMatchPattern1, &h_initialMatch, sizeof(FoundMatchInfo), cudaMemcpyHostToDevice));

	dim3 block(TPB_1D, TPB_1D);
	// Grid dimensions cover every possible top-left pixel of the source image
	dim3 grid_src((gl.imageWidth + block.x - 1) / block.x, (gl.imageHeight + block.y - 1) / block.y);

	// --- Match Pattern 0 ---
	if (pattern0_data.texObj != 0 && pattern0_data.width > 0 && pattern0_data.height > 0) {
		printf("Launching kernel for pattern 0 (%u x %u). Source: %u x %u\n", pattern0_data.width, pattern0_data.height, gl.imageWidth, gl.imageHeight);
		patternMatchKernel << <grid_src, block >> > (
			cd_src.texObj, gl.imageWidth, gl.imageHeight,
			pattern0_data.texObj, pattern0_data.width, pattern0_data.height,
			d_foundMatchPattern0
			);
		checkCudaErrors(cudaGetLastError());
	}
	else {
		printf("Pattern 0 not loaded or invalid, skipping match.\n");
	}

	// --- Match Pattern 1 ---
	if (pattern1_data.texObj != 0 && pattern1_data.width > 0 && pattern1_data.height > 0) {
		printf("Launching kernel for pattern 1 (%u x %u). Source: %u x %u\n", pattern1_data.width, pattern1_data.height, gl.imageWidth, gl.imageHeight);
		patternMatchKernel << <grid_src, block >> > (
			cd_src.texObj, gl.imageWidth, gl.imageHeight,
			pattern1_data.texObj, pattern1_data.width, pattern1_data.height,
			d_foundMatchPattern1
			);
		checkCudaErrors(cudaGetLastError());
	}
	else {
		printf("Pattern 1 not loaded or invalid, skipping match.\n");
	}

	checkCudaErrors(cudaDeviceSynchronize()); // Wait for all kernels to complete

	// Copy results from GPU to CPU
	FoundMatchInfo h_matchPattern0, h_matchPattern1;
	checkCudaErrors(cudaMemcpy(&h_matchPattern0, d_foundMatchPattern0, sizeof(FoundMatchInfo), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&h_matchPattern1, d_foundMatchPattern1, sizeof(FoundMatchInfo), cudaMemcpyDeviceToHost));

	// Print results
	printf("\n--- Pattern Matching Results ---\n");
	if (h_matchPattern0.x != -1) {
		printf("Pattern 0 (%s) FOUND at: (%d, %d) with rotation index: %d (0:0, 1:90, 2:180, 3:270)\n",
			PATTERN0_IMG_FN, h_matchPattern0.x, h_matchPattern0.y, h_matchPattern0.rotation_idx);
	}
	else {
		printf("Pattern 0 (%s) NOT FOUND.\n", PATTERN0_IMG_FN);
	}

	if (h_matchPattern1.x != -1) {
		printf("Pattern 1 (%s) FOUND at: (%d, %d) with rotation index: %d (0:0, 1:90, 2:180, 3:270)\n",
			PATTERN1_IMG_FN, h_matchPattern1.x, h_matchPattern1.y, h_matchPattern1.rotation_idx);
	}
	else {
		printf("Pattern 1 (%s) NOT FOUND.\n", PATTERN1_IMG_FN);
	}
	printf("-------------------------------\n");

	matchExecutionDone = true;
	//glutIdleFunc(NULL); // Stop calling idle function if we only want to run once
}

void initCUDAObjects()
{
	checkCudaErrors(cudaGraphicsGLRegisterImage(&cd_src.texResource, gl.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
	checkCudaErrors(cudaGraphicsMapResources(1, &cd_src.texResource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cd_src.texArrayData, cd_src.texResource, 0, 0));

	cd_src.resDesc.resType = cudaResourceTypeArray;
	cd_src.resDesc.res.array.array = cd_src.texArrayData;

	cd_src.texDesc.addressMode[0] = cudaAddressModeClamp;
	cd_src.texDesc.addressMode[1] = cudaAddressModeClamp;
	cd_src.texDesc.filterMode = cudaFilterModePoint;
	cd_src.texDesc.readMode = cudaReadModeElementType;
	cd_src.texDesc.normalizedCoords = false;

	checkCudaErrors(cudaGetChannelDesc(&cd_src.texChannelDesc, cd_src.texArrayData));

	checkCudaErrors(cudaCreateTextureObject(&cd_src.texObj, &cd_src.resDesc, &cd_src.texDesc, NULL));

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cd_src.texResource, 0));

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cd_src.pboResource, gl.pboID, cudaGraphicsMapFlagsWriteDiscard));


	pattern0_data.loadPattern(PATTERN0_IMG_FN);
	pattern1_data.loadPattern(PATTERN1_IMG_FN);

	checkCudaErrors(cudaMalloc((void**)&d_foundMatchPattern0, sizeof(FoundMatchInfo)));
	checkCudaErrors(cudaMalloc((void**)&d_foundMatchPattern1, sizeof(FoundMatchInfo)));

}

void releaseCUDA()
{
	// Release source image CUDA objects
	if (cd_src.pboResource) cudaGraphicsUnregisterResource(cd_src.pboResource);
	cd_src.pboResource = nullptr;
	if (cd_src.texResource) cudaGraphicsUnregisterResource(cd_src.texResource);
	cd_src.texResource = nullptr;
	if (cd_src.texObj) cudaDestroyTextureObject(cd_src.texObj); // texArrayData is part of texResource, not freed separately
	cd_src.texObj = 0;
	cd_src.texArrayData = nullptr;


	// Release pattern image CUDA objects
	pattern0_data.release();
	pattern1_data.release();

	// Release result memory
	if (d_foundMatchPattern0) cudaFree(d_foundMatchPattern0);
	d_foundMatchPattern0 = nullptr;
	if (d_foundMatchPattern1) cudaFree(d_foundMatchPattern1);
	d_foundMatchPattern1 = nullptr;

	cudaDeviceReset();
}
#pragma endregion

#pragma region OpenGL Routines

// Load image, prepare GL Texture and PBO (for source image display)
void prepareGlObjects(const char* imageFileName)
{
	printf("Loading image: %s\n", imageFileName);
	FIBITMAP* tmp = ImageManager::GenericLoader(imageFileName, 0);
	if (!tmp) {
		fprintf(stderr, "ERROR: Could not load image %s\n", imageFileName);
		exit(EXIT_FAILURE);
	}

	gl.imageWidth = FreeImage_GetWidth(tmp);
	gl.imageHeight = FreeImage_GetHeight(tmp);
	gl.imageBPP = FreeImage_GetBPP(tmp);
	gl.imagePitch = FreeImage_GetPitch(tmp);

	// Ensure image is 32-bit for RGBA consistency
	if (gl.imageBPP != 32) {
		printf("Converting image to 32bpp...\n");
		FIBITMAP* temp = FreeImage_ConvertTo32Bits(tmp);
		if (!temp) {
			fprintf(stderr, "ERROR: FreeImage_ConvertTo32Bits failed!\n");
			FreeImage_Unload(tmp);
			exit(EXIT_FAILURE);
		}
		FreeImage_Unload(tmp);
		tmp = temp;
		gl.imageBPP = FreeImage_GetBPP(tmp);
		gl.imagePitch = FreeImage_GetPitch(tmp);
		if (gl.imageBPP != 32) { // Sanity check
			fprintf(stderr, "ERROR: Image BPP is %u after conversion attempt!\n", gl.imageBPP);
			FreeImage_Unload(tmp);
			exit(EXIT_FAILURE);
		}
	}

	// Determine source format for glTexImage2D based on masks
	unsigned int red_mask = FreeImage_GetRedMask(tmp);
	unsigned int green_mask = FreeImage_GetGreenMask(tmp);
	unsigned int blue_mask = FreeImage_GetBlueMask(tmp);
	GLenum sourceFormat = GL_BGRA; // Common default for 32bpp loaded by FreeImage on Windows
	if (red_mask == 0x000000FF && green_mask == 0x0000FF00 && blue_mask == 0x00FF0000) {
		printf("Detected RGBA byte order from FreeImage.\n");
		sourceFormat = GL_RGBA;
	}
	else if (red_mask == 0x00FF0000 && green_mask == 0x0000FF00 && blue_mask == 0x000000FF) {
		printf("Detected BGRA byte order from FreeImage.\n");
		sourceFormat = GL_BGRA;
	}
	else {
		printf("Warning: Unusual 32bpp color mask order. Assuming BGRA.\n");
	}

	glGenTextures(1, &gl.textureID);
	glBindTexture(GL_TEXTURE_2D, gl.textureID);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, gl.imagePitch / (gl.imageBPP / 8)); // Use pitch info

	BYTE* pixelData = FreeImage_GetBits(tmp);
	if (!pixelData) {
		fprintf(stderr, "ERROR: FreeImage_GetBits returned NULL!\n");
		FreeImage_Unload(tmp);
		exit(EXIT_FAILURE);
	}

	// Upload image data to GPU Texture. Use GL_RGBA8 for internal format for consistency.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gl.imageWidth, gl.imageHeight, 0, sourceFormat, GL_UNSIGNED_BYTE, pixelData);
	GLenum glError = glGetError();
	if (glError != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL Error after glTexImage2D: 0x%x\n", glError);
		FreeImage_Unload(tmp);
		exit(EXIT_FAILURE);
	}

	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0); // Reset row length

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	FreeImage_Unload(tmp); // Free host bitmap memory

	glGenBuffers(1, &gl.pboID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl.pboID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, (size_t)gl.imageWidth * gl.imageHeight * 4, NULL, GL_DYNAMIC_DRAW); // Allocate PBO size
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	printf("OpenGL objects prepared.\n");
}


void my_display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, gl.textureID); // Display the source image

	glBegin(GL_QUADS);
	glTexCoord2d(0, 0); glVertex2d(0, 0);
	glTexCoord2d(1, 0); glVertex2d(gl.viewportWidth, 0);
	glTexCoord2d(1, 1); glVertex2d(gl.viewportWidth, gl.viewportHeight);
	glTexCoord2d(0, 1); glVertex2d(0, gl.viewportHeight);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glFlush();
	glutSwapBuffers();
}

// GLUT resize callback
void my_resize(GLsizei w, GLsizei h)
{
	gl.viewportWidth = w;
	gl.viewportHeight = h;
	glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, gl.viewportWidth, 0, gl.viewportHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

// GLUT idle callback
void my_idle()
{
	if (!matchExecutionDone) {
		cudaPatternMatchWorker(); // Perform pattern matching
		// cudaPatternMatchWorker sets matchExecutionDone = true
		// and can call glutIdleFunc(NULL) itself if desired.
		if (matchExecutionDone) {
			printf("\nPattern matching complete. Displaying source image. Close window to exit.\n");
		}
	}
	// If we want continuous display updates even after matching:
	glutPostRedisplay();
}

// Initialize GLUT and OpenGL
void initGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(gl.viewportWidth, gl.viewportHeight);
	glutInitWindowPosition(100, 100);
	glutSetOption(GLUT_RENDERING_CONTEXT, GLUT_CREATE_NEW_CONTEXT);
	glutCreateWindow("CUDA Rotated Pattern Matching");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0")) {
		fprintf(stderr, "OpenGL 2.0 not available\n");
		exit(EXIT_FAILURE);
	}

	glutDisplayFunc(my_display);
	glutReshapeFunc(my_resize);
	glutIdleFunc(my_idle);
	//glutKeyboardFunc(my_keyboard); // Optional: for user interaction

	glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Dark gray background
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, gl.viewportWidth, gl.viewportHeight);
	glFlush();
}

// Release OpenGL resources
void releaseOpenGL()
{
	if (gl.textureID > 0) glDeleteTextures(1, &gl.textureID);
	gl.textureID = 0;
	if (gl.pboID > 0) glDeleteBuffers(1, &gl.pboID);
	gl.pboID = 0;
}
#pragma endregion OpenGL Routines


bool resourcesReleased = false;
void releaseResources() // atexit callback
{
	if (!resourcesReleased) {
		printf("\nReleasing resources...\n");
		// CUDA resources should be released before GL context is destroyed if interop involved
		// However, glutMainLoop might destroy context before atexit.
		// Better to call releaseCUDA explicitly before exiting glutMainLoop if possible,
		// or ensure no GL context dependency in releaseCUDA for objects already unregistered.
		releaseCUDA();
		releaseOpenGL();
		FreeImage_DeInitialise();
		printf("Resources released.\n");
		resourcesReleased = true;
	}
}


int main(int argc, char* argv[])
{
	printf("Starting CUDA Rotated Pattern Matching program...\n");
	FreeImage_Initialise();
	atexit(releaseResources);

	int dev = findCudaDevice(argc, (const char**)argv);
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using CUDA device %d: %s\n", dev, deviceProp.name);
	checkCudaErrors(cudaSetDevice(dev)); // Ensure device is set

	initGL(argc, argv); // Initialize OpenGL and GLUT

	prepareGlObjects(SOURCE_IMG_FN);

	initCUDAObjects();

	printf("\nOpenGL and CUDA initialized. Starting pattern matching via GLUT idle function.\n");
	printf("Results will be printed to console.\n");

	glutMainLoop();

	printf("\nExiting application.\n");
	// releaseResources() will be called by atexit
	return EXIT_SUCCESS;
}