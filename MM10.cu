// C includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

#define MAX_EPSILON_ERROR 5.0f

const char *sSDKsample = "Filtre Sobel";
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = NULL;

bool g_bQAReadback = false;

unsigned char *pixels = NULL;  // Image pixel data on the host
float imageScale = 1.f;        // Image exposure

int *pArgc   = NULL;
char **pArgv = NULL;

void runAutoTest(int argc, char **argv);

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)

typedef unsigned char Pixel; // In the .h
#include <helper_string.h>

__global__ void SobelTex(Pixel *pSobelOriginal, unsigned int Pitch, int w, int h, float fScale) {
    unsigned char *pSobel = (unsigned char *)(((char *) pSobelOriginal)+blockIdx.x*Pitch);

    for (int i = threadIdx.x; i < w; i += blockDim.x)
    {
        unsigned char pix00 = tex2D(tex, (float) i-1, (float) blockIdx.x-1);
        unsigned char pix01 = tex2D(tex, (float) i+0, (float) blockIdx.x-1);
        unsigned char pix02 = tex2D(tex, (float) i+1, (float) blockIdx.x-1);
        unsigned char pix10 = tex2D(tex, (float) i-1, (float) blockIdx.x+0);
        unsigned char pix11 = tex2D(tex, (float) i+0, (float) blockIdx.x+0);
        unsigned char pix12 = tex2D(tex, (float) i+1, (float) blockIdx.x+0);
        unsigned char pix20 = tex2D(tex, (float) i-1, (float) blockIdx.x+1);
        unsigned char pix21 = tex2D(tex, (float) i+0, (float) blockIdx.x+1);
        unsigned char pix22 = tex2D(tex, (float) i+1, (float) blockIdx.x+1);
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale);
    }
}

// Wrapper for the __global__ call that sets up the texture and threads
void sobelFilter(Pixel *odata, int iw, int ih, float fScale) {
		SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale);
    checkCudaErrors(cudaUnbindTexture(tex));
}

void initializeData(char *file) {
    GLint bsize;
    unsigned int w, h;
    size_t file_length = strlen(file);

    if (!strcmp(&file[file_length-3], "pgm")) {
        if (sdkLoadPGM<unsigned char>(file, &pixels, &w, &h) != true) {
            printf("Failed to load PGM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else if (!strcmp(&file[file_length-3], "ppm")) {
        if (sdkLoadPPM4(file, &pixels, &w, &h) != true) {
            printf("Failed to load PPM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else {
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    imWidth = (int)w;
    imHeight = (int)h;
}

void loadDefaultImage(char *loc_exec) {
    printf("Reading image: lena.pgm\n");
    const char *image_filename = "lena.pgm";
    char *image_path = sdkFindFilePath(image_filename, loc_exec);

    if (image_path == NULL) {
        printf("Failed to read image file: <%s>\n", image_filename);
        exit(EXIT_FAILURE);
    }

    initializeData(image_path);
    free(image_path);
}

void runAutoTest(int argc, char *argv[]) {
    printf("[%s] (automated testing w/ readback)\n", sSDKsample);
    int devID = findCudaDevice(argc, (const char **)argv);

    loadDefaultImage(argv[0]);

    Pixel *d_result;
    checkCudaErrors(cudaMalloc((void **)&d_result, imWidth*imHeight*sizeof(Pixel)));

    char *ref_file = NULL;
    char  dump_file[256];

    getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);

    printf("AutoTest: %s\n", sSDKsample);
    sobelFilter(d_result, imWidth, imHeight, imageScale);
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char *h_result = (unsigned char *)malloc(imWidth*imHeight*sizeof(Pixel));
    checkCudaErrors(cudaMemcpy(h_result, d_result, imWidth*imHeight*sizeof(Pixel), cudaMemcpyDeviceToHost));
    sdkSavePGM(dump_file, h_result, imWidth, imHeight);

    if (!sdkComparePGM(dump_file, sdkFindFilePath(ref_file, argv[0]), MAX_EPSILON_ERROR, 0.15f, false))
    {
        g_TotalErrors++;
    }

    checkCudaErrors(cudaFree(d_result));
    free(h_result);

    if (g_TotalErrors != 0)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed!\n");
    exit(EXIT_SUCCESS);
}


int main(int argc, char **argv) {
		pArgc = &argc;
		pArgv = argv;

    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        printf("\nUsage: SobelFilter <options>\n");
				printf("\t\t-mode=n (0=original, 1=texture, 2=smem + texture)\n");
        printf("\t\t-file=ref_orig.pgm (ref_tex.pgm, ref_shared.pgm)\n\n");
				printf("\t\t-device=n (n=0,1,2,...)\n");
        exit(EXIT_SUCCESS);
    }
		if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
        g_bQAReadback = true;
        runAutoTest(argc, argv);
    }

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        printf("   This SDK does not explicitly support -device=n when running with OpenGL.\n");
        printf("   When specifying -device=n (n=0,1,2,....) the sample must not use OpenGL.\n");
        printf("   See details below to run without OpenGL:\n\n");
        printf(" > %s -device=n\n\n", argv[0]);
        printf("exiting...\n");
        exit(EXIT_SUCCESS);
    }

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    loadDefaultImage(argv[0]);

    // If code is not printing the usage, then we execute this path.
    printf("I: display Image (no filtering)\n");
    printf("T: display Sobel Edge Detection (Using Texture)\n");
    printf("S: display Sobel Edge Detection (Using SMEM+Texture)\n");
    printf("Use the '-' and '=' keys to change the brightness.\n");
    fflush(stdout);

#if defined (__APPLE__) || defined(MACOSX)
    atexit(cleanup);
#else
    glutCloseFunc(cleanup);
#endif

    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMainLoop();
}



















#define SIZE 32

#ifndef PINNED
#define PINNED 0
#endif


// Matriz por Matriz
// C(NxM) <- A(NxP) * B (PxM)

__global__ void Kernel10(int N, int M, int P, float *A, float *B, float *C) {

  __shared__ float sA[SIZE][SIZE];
  __shared__ float sB[SIZE][SIZE];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row = by * SIZE + ty;
  int col = bx * SIZE + tx;

  float tmp = 0.0;
  for (int m=0; m < P; m=m+SIZE) {
    sA[ty][tx] = A[row*P + m + tx];
    sB[ty][tx] = B[col + (m + ty)*M];
    __syncthreads();

    for (int k=0; k<SIZE; k++)
      tmp += sA[ty][k] * sB[k][tx];

    __syncthreads();
  }
  C[row*M+col] = tmp;
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
  if (argc == 5) {
     N = atoi(argv[1]);
     M = atoi(argv[2]);
     P = atoi(argv[3]);
     test = *argv[4];
  }
  else { printf("Usage: ./exe N M P test\n"); exit(0); }

  // numero de Threads en cada dimension
  nThreads = SIZE;

  // numero de Blocks en cada dimension
  nBlocksN = (N+nThreads-1)/nThreads;
  nBlocksM = (M+nThreads-1)/nThreads;

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
    h_A = (float*) malloc(numBytesA);
    h_B = (float*) malloc(numBytesB);
    h_C = (float*) malloc(numBytesC);
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
  Kernel10<<<dimGrid, dimBlock>>>(N, M, P, d_A, d_B, d_C);

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

  cudaEventElapsedTime(&TiempoTotal,  E0, E3);
  cudaEventElapsedTime(&TiempoKernel, E1, E2);
  printf("\nKERNEL 10\n");
  printf("Dimensiones: %dx%d <- %dx%d * %dx%d\n", N, M, N, P, P, M);
  printf("nThreads: %dx%d (%d)\n", nThreads, nThreads, nThreads * nThreads);
  printf("nBlocks: %dx%d (%d)\n", nBlocksM, nBlocksN, nBlocksN*nBlocksM);
  if (PINNED) printf("Usando Pinned Memory\n");
         else printf("NO usa Pinned Memory\n");
  printf("Tiempo Global: %4.6f milseg\n", TiempoTotal);
  printf("Tiempo Kernel: %4.6f milseg\n", TiempoKernel);
  printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoTotal));
  printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 * (float) N * (float) M * (float) P) / (1000000.0 * TiempoKernel));

  cudaEventDestroy(E0); cudaEventDestroy(E1); cudaEventDestroy(E2); cudaEventDestroy(E3);

  if (test == 'N')
    printf ("NO TEST\n");
  else  if (TestMM(N, M, P, h_A, h_B, h_C))
    printf ("TEST PASS\n");
  else
    printf ("TEST FAIL\n");

  if (PINNED) {
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
  }
  else {
    free(h_A); free(h_B); free(h_C);
  }

}


void InitM(int N, int M, float *Mat) {
   int i;
   for (i=0; i<N*M; i++)
     Mat[i] = rand() / (float) RAND_MAX;

}

int error(float a, float b) {
  float tmp;

  tmp = abs(a-b) / abs(min(a,b));

  if (tmp > 0.0001) return 1;
  else  return 0;

}

int TestMM(int N, int M, int P, float *A, float *B, float *C) {
   int i, j, k;
   float tmp;
   for (i=0; i<N; i++)
     for (j=0; j<M; j++) {
       tmp = 0.0;
       for (k=0; k<P; k++)
         tmp = tmp + A[i*P+k] * B[k*M+j];
       if (error(tmp, C[i*M+j])) {
         printf ("%d:%d: %f - %f = %f \n", i, j, tmp, C[i*M+j], abs(tmp - C[i*M+j]));
         return 0;
       }
     }

   return 1;
}
