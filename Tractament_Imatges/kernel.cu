#include <stdio.h>
#include <stdlib.h>

#define SIZE 32

#ifndef PINNED
#define PINNED 0
#endif


// Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)

__global__ void Kernel01(int N, int M, int P, float *A, float *B, float *C) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N && col < M) {
		float tmp = 0.0;
		for (int k = 0; k < P; k++)
			tmp += A[row*P + k] * B[k*M + col];
		C[row*M + col] = tmp;
	}
}



void InitM(int N, int M, float *Mat);
int TestMM(int N, int M, int P, float *A, float *B, float *C);


// Invocacion:
// ./ejecutable N M P test
// TAM es el la dimension de las matrices
// test == 'Y', comprueba que el resultado sea correcto
// test == 'N', NO comprueba que el resultado (Util para tomar tiempos)
// Por defecto, N = 639, M = 641, P = 1023, test == 'N'

int main(int argc, char** argv)
{
	unsigned int N, M, P;
	unsigned int numBytesC, numBytesA, numBytesB;
	unsigned int nBlocksN, nBlocksM, nThreads;

	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;

	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	char test;

	// Dimension de las matrices NxM, NxP, PxM y comprobacion resultado
	
	N = 64;
	M = 1024;
	P = 128;
	test = 'Y';
	//else { printf("Usage: ./exe N M P test\n"); exit(0); }

	// numero de Threads en cada dimension 
	nThreads = SIZE;

	// numero de Blocks en cada dimension 
	nBlocksN = (N + nThreads - 1) / nThreads;
	nBlocksM = (M + nThreads - 1) / nThreads;

	numBytesC = N * M * sizeof(float);
	numBytesA = N * P * sizeof(float);
	numBytesB = P * M * sizeof(float);

	dim3 dimGrid(nBlocksM, nBlocksN, 1);
	dim3 dimBlock(nThreads, nThreads, 1);

	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);

	if (PINNED) {
		// Obtiene Memoria [pinned] en el host
		cudaMallocHost((float**)&h_A, numBytesA);
		cudaMallocHost((float**)&h_B, numBytesB);
		cudaMallocHost((float**)&h_C, numBytesC);
	}
	else {
		// Obtener Memoria en el host
		h_A = (float*)malloc(numBytesA);
		h_B = (float*)malloc(numBytesB);
		h_C = (float*)malloc(numBytesC);
	}

	// Inicializa las matrices
	InitM(N, P, h_A);
	InitM(P, M, h_B);

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Obtener Memoria en el device
	cudaMalloc((float**)&d_A, numBytesA);
	cudaMalloc((float**)&d_B, numBytesB);
	cudaMalloc((float**)&d_C, numBytesC);

	// Copiar datos desde el host en el device 
	cudaMemcpy(d_A, h_A, numBytesA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numBytesB, cudaMemcpyHostToDevice);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Ejecutar el kernel 
	Kernel01 << <dimGrid, dimBlock >> > (N, M, P, d_A, d_B, d_C);

	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Obtener el resultado desde el host 
	cudaMemcpy(h_C, d_C, numBytesC, cudaMemcpyDeviceToHost);

	// Liberar Memoria del device 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);

	cudaEventElapsedTime(&TiempoTotal, E0, E3);
	cudaEventElapsedTime(&TiempoKernel, E1, E2);
	printf("\nKERNEL 01\n");
	printf("Dimensiones: %dx%d <- %dx%d * %dx%d\n", N, M, N, P, P, M);
	printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
	printf("nBlocks: %dx%d (%d)\n", nBlocksM, nBlocksN, nBlocksN*nBlocksM);
	if (PINNED) printf("Usando Pinned Memory\n");
	else printf("NO usa Pinned Memory\n");
	printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
	printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
	printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float)N * (float)M * (float)P) / (1000000.0 * TiempoTotal));
	printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float)N * (float)M * (float)P) / (1000000.0 * TiempoKernel));

	cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

	if (test == 'N')
		printf("NO TEST\n");
	else  if (TestMM(N, M, P, h_A, h_B, h_C))
		printf("TEST PASS\n");
	else
		printf("TEST FAIL\n");

	if (PINNED) {
		cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
	}
	else {
		free(h_A); free(h_B); free(h_C);
	}

}


void InitM(int N, int M, float *Mat) {
	int i;
	for (i = 0; i < N*M; i++)
		Mat[i] = rand() / (float)RAND_MAX;

}

int error(float a, float b) {
	float tmp;

	tmp = abs(a - b) / abs(min(a, b));

	if (tmp > 0.0001) return 1;
	else  return 0;

}

int TestMM(int N, int M, int P, float *A, float *B, float *C) {
	int i, j, k;
	float tmp;
	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++) {
			tmp = 0.0;
			for (k = 0; k < P; k++)
				tmp = tmp + A[i*P + k] * B[k*M + j];
			if (error(tmp, C[i*M + j])) {
				printf("%d:%d: %f - %f = %f \n", i, j, tmp, C[i*M + j], abs(tmp - C[i*M + j]));
				return 0;
			}
		}

	return 1;
}

