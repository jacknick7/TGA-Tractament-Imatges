#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cutil.h>
#include <ctime>
/* Includes per CUDA a Windows */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

unsigned int width, height;

int mask[3][3] = {	1, 2, 1,
					2, 3, 2,
					1, 2, 1,
					};

int getPixel(unsigned char * arr, int col, int row) {
	int sum = 0;
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			int color = arr[(row + j) * width + (col + i)];
			sum += color * mask[i + 1][j + 1];
		}
	}
	return sum / 15;
}

void h_blur(unsigned char * arr, unsigned char * result) {
	int offset = 2 * width;
	for (int row = 2; row < height - 3; row++) {
		for (int col = 2; col < width - 3; col++) {
			result[offset + col] = getPixel(arr, col, row);
		}
		offset += width;
	}
}

__global__ void d_blur(unsigned char * arr, unsigned char * result, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < 2 || col < 2 || row >= height - 3 || col >= width - 3)
		return;
	int mask[3][3] = { 1 ,2 ,1 , 2 ,3 ,2 , 1 ,2 ,1 };
	int sum = 0;
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			int color = arr[(row + j) * width + (col + i)];
			sum += color * mask[i + 1][j + 1];
		}
	}
	result[row * width + col] = sum / 15;
}

int main(int argc, char ** argv) {

	unsigned char * d_resultPixels;
	unsigned char * h_resultPixels;
	unsigned char * h_pixels = NULL;
	unsigned char * d_pixels = NULL;

	char * srcPath = "/ Developer /GPU Computing /C/src / GaussianBlur /image / wallpaper2.pgm";
	char * h_ResultPath = "/ Developer /GPU Computing /C/src /GaussianBlur / output / h_wallpaper2.pgm ";
	char * d_ResultPath = "/ Developer /GPU Computing /C/src /GaussianBlur / output / d_wallpaper2.pgm ";
	
	cutLoadPGMub(srcPath, &h_pixels, &width, &height);
	
	int ImageSize = sizeof(unsigned char) * width * height;

	h_resultPixels = (unsigned char *)malloc(ImageSize);
	cudaMalloc((void **)& d_pixels, ImageSize);
	cudaMalloc((void **)& d_resultPixels, ImageSize);
	cudaMemcpy(d_pixels, h_pixels, ImageSize, cudaMemcpyHostToDevice);
	
	/* **** CPU **** */
	
	clock_t starttime, endtime, difference;
	starttime = clock();

	// blur gaussia
	h_blur(h_pixels, h_resultPixels);

	endtime = clock();
	difference = (endtime - starttime);
	double interval = difference / (double)CLOCKS_PER_SEC;
	printf("CPU execution time = %f ms\n", interval * 1000);

	cutSavePGMub(h_ResultPath, h_resultPixels, width, height); // Sale Mal

	/* **** CPU **** */
	
	/* **** GPU **** */
	
	dim3 block(16, 16);
	dim3 grid(width / 16, height / 16);
	unsigned int timer = 0;
	// cutCreateTimer(&timer);
	// cutStartTimer(timer);

	/* CUDA method */
	d_blur << < grid, block > > > (d_pixels, d_resultPixels, width, height);
	
	cudaThreadSynchronize();
	// cutStopTimer(timer);
	// printf(" CUDA execution time = %f ms\n", cutGetTimerValue(timer));
	
	cudaMemcpy(h_resultPixels, d_resultPixels, ImageSize, cudaMemcpyDeviceToHost);
	cutSavePGMub(d_ResultPath, h_resultPixels, width, height);

	/* **** GPU **** */
	
	printf(" Press enter to exit ...\ n");
	getchar();
}
