// C includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking


__device__ unsigned char ComputeSobel(unsigned char ul, 	// upper left
																			unsigned char um, 	// upper middle
																			unsigned char ur, 	// upper right
																			unsigned char ml, 	// middle left
																			unsigned char mm, 	// middle (unused)
																			unsigned char mr, 	// middle right
																			unsigned char ll, 	// lower left
																			unsigned char lm, 	// lower middle
																			unsigned char lr) {	// lower right
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short)(fScale*(abs((int)Horz)+abs((int)Vert)));

    if (Sum < 0) return 0;
    else if (Sum > 0xff) return 0xff;

    return (unsigned char) Sum;
}

__global__ void SobelFilter(int w, int h, const unsigned char *d_imageIn, unsigned char *d_imageOut) {
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        d_imageOut[i] = (char) ((int) d_imageIn[i] / 4);
    }
}

int main(int argc, char** argv) {
		unsigned int numBytesIn, numBytesOut;

		float TotalTime, KernelTime;
		cudaEvent_t E0, E1, E2, E3;

		unsigned char *h_imageIn, *h_imageOut;
		unsigned char *d_imageIn, *d_imageOut;

		char *imageIn_path = "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/3_Imaging/SobelFilter/data/lena.pgm";
		char *imageOut_path = "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/3_Imaging/SobelFilter/data/lena_low.pgm";

		if (argc > 3) {
				printf("Usage: ./SobelFilter path_in path_out\n");
				exit(0);
		}
		if (argc > 1) {
				imageIn_path = argv[1];
				if (argc > 2) imageOut_path = argv[2];
				printf("Reading custom image\n");
		}
		else printf("Reading image: lena.pgm\n");

		unsigned int w, h;
		if (sdkLoadPGM<unsigned char>(imageIn_path, &h_imageIn, &w, &h) != true) {
				printf("Failed to load PGM image file: %s\n", image_path);
				exit(EXIT_FAILURE);
		}
		int imWidth = (int) w;
		int imHeight = (int) h;

		numBytesIn = imWidth * imHeight * sizeof(unsigned char);
		numBytesOut = imWidth * imHeight * sizeof(unsigned char);

		cudaEventCreate(&E0);
		cudaEventCreate(&E1);
		cudaEventCreate(&E2);
		cudaEventCreate(&E3);

		// Obtener Memoria en el host
		h_imageOut = (unsigned char*) malloc(numBytesA);

		cudaEventRecord(E0, 0);
		cudaEventSynchronize(E0);

		// Obtener Memoria en el device
		cudaMalloc((unsigned char**)&d_imageIn, numBytesIn);
		cudaMalloc((unsigned char**)&d_imageOut, numBytesOut);

		// Copiar datos desde el host en el device
		cudaMemcpy(d_imageIn, h_imageIn, numBytesIn, cudaMemcpyHostToDevice);

		cudaEventRecord(E1, 0);
		cudaEventSynchronize(E1);

		// Ejecutar el kernel
		SobelFilter<<<imHeight, 384>>>(imWidth, imHeight, d_imageIn, d_imageOut);

		cudaEventRecord(E2, 0);
		cudaEventSynchronize(E2);

		// Obtener el resultado desde el host
		cudaMemcpy(h_imageOut, d_imageOut, numBytesOut, cudaMemcpyDeviceToHost);

		// Liberar Memoria del device
		cudaFree(d_imageIn);
		cudaFree(d_imageOut);

		cudaEventRecord(E3, 0);
		cudaEventSynchronize(E3);

		cudaEventElapsedTime(&TotalTime,  E0, E3);
		cudaEventElapsedTime(&KernelTime, E1, E2);
		printf("\nSobel Filter\n");
		printf("Global Time: %4.6f ms\n", TotalTime);
		printf("Kernel Time: %4.6f ms\n", KernelTime);

		cudaEventDestroy(E0); cudaEventDestroy(E1);
		cudaEventDestroy(E2); cudaEventDestroy(E3);

		sdkSavePGM(imageOut_path, h_imageOut, imWidth, imHeight);

		free(h_imageIn); free(h_imageOut);

}
