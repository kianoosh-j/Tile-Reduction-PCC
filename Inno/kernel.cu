
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define length 80000
#define step 32 //  DON'T CHANGE THIS. SHOULD BE THE SAME AS blockDim.x
#define width 192
#define height 256
#define depth 16


cudaError_t computeCorrCoefOnDevice(float *c, const int *a, const int *b, int sz);
__global__ void corrcoef_kernel_v0(int *a_dev, int *b_dev, float *c_dev);
__global__ void corrcoef_kernel_v1(int *a_dev, int *b_dev, float *c_dev);
__global__ void corrcoef_kernel_v2(int *a_dev, int *b_dev, float *c_dev);
void checkResult(int *a, int *b, float *c);


__global__ void corrcoef_kernel_v0(int *a_dev, int *b_dev, float *c_dev)
{
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	if (row >= height || col >= width)
		return;

	int   exy = 0, ex = 0, ex2 = 0, ey = 0, ey2 = 0;
	float fexy = 0, fex = 0, fex2 = 0, fey = 0, fey2 = 0;

	for (int iter = 0; iter < length; iter++)
	{
		int a = a_dev[row * length + iter];
		int b = b_dev[iter * width + col];

		ex += a;
		ex2 += a * a;
		ey += b;
		ey2 += b * b;
		exy += a * b;
	}

	fexy = (float)((float)exy / (float)length);//20000.0;//length;
	fex = (float)((float)ex / (float)length);//20000.0;
	fex2 = (float)((float)ex2 / (float)length);//20000.0;
	fey = (float)((float)ey / (float)length);//20000.0;
	fey2 = (float)((float)ey2 / (float)length);//20000.0;

	float covXY = fexy - fex * fey;
	float varX = fex2 - fex * fex;
	float varY = fey2 - fey * fey;

	c_dev[row * width + col] = covXY / sqrt(varX * varY);
}


__global__ void corrcoef_kernel_v1(int *a_dev, int *b_dev, float *c_dev)
{
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	if (row >= height || col >= width)
		return;

	int   exy = 0, ex = 0, ex2 = 0, ey = 0, ey2 = 0;
	float fexy = 0, fex = 0, fex2 = 0, fey = 0, fey2 = 0;
	__shared__ int a_shared[32][step];
	__shared__ int b_shared[step][16];

	// MAIN LOOP OF THE PROGRAM
	for (int iter = 0 ; iter < length ; iter += step) {
		a_shared[ty][tx] = a_dev[row * length + (tx + iter)];
		a_shared[ty][tx + blockDim.x] = a_dev[row * length + (tx + blockDim.x + iter)];
		b_shared[ty][tx] = b_dev[col + (ty + iter) * width];
		__syncthreads();

		// INNER MOST LOOP OVER SHARED MEMORY: COMPUTES ALL THE NECESSERY ELEMENTS i.e. first moment of a_dev
		for (int iter2 = 0; iter2 < step; iter2++) {
			ex += a_shared[ty][iter2];
			ex2 += a_shared[ty][iter2] * a_shared[ty][iter2];
			ey += b_shared[iter2][tx];
			ey2 += b_shared[iter2][tx] * b_shared[iter2][tx];
			exy += a_shared[ty][iter2] * b_shared[iter2][tx];
		}
			
		__syncthreads();
	}

	fexy = (float)((float)exy / (float)length);//20000.0;//length;
	fex = (float)((float)ex / (float)length);//20000.0;
	fex2 = (float)((float)ex2 / (float)length);//20000.0;
	fey = (float)((float)ey / (float)length);//20000.0;
	fey2 = (float)((float)ey2 / (float)length);//20000.0;

	float covXY = fexy - fex * fey;
	float varX = fex2 - fex * fex;
	float varY = fey2 - fey * fey;


	c_dev[row * width + col] = covXY / sqrt(varX * varY);
}



__global__ void corrcoef_kernel_v2(int *a_dev, int *b_dev, float *c_dev)
{
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;

	if (row >= height || col >= width)
		return;

	int   exy = 0, ex = 0, ex2 = 0, ey = 0, ey2 = 0;
	float fexy = 0, fex = 0, fex2 = 0, fey = 0, fey2 = 0;
	__shared__ int a_shared[32][step];
	__shared__ int b_shared[step][16];

	// MAIN LOOP OF THE PROGRAM
	for (int iter = 0; iter < length; iter += step) {
		a_shared[ty][tx] = a_dev[row*length + (tx + iter)];
		a_shared[ty][tx + blockDim.x] = a_dev[row*length + (tx + blockDim.x + iter)];
		b_shared[ty][tx] = b_dev[col + (ty + iter)*width];
		__syncthreads();

		// MY IDEA: FIRST AND SECOND MOMENETS COMPUTES OUT OF THE INNER MOST LOOP
		ex += a_shared[ty][tx];
		ex += a_shared[ty][tx + blockDim.x];

		ex2 += a_shared[ty][tx] * a_shared[ty][tx];
		ex2 += a_shared[ty][tx + blockDim.x] * a_shared[ty][tx + blockDim.x];

		ey += b_shared[ty][tx];
		ey2 += b_shared[ty][tx] * b_shared[ty][tx];

		// INNER MOST LOOP OVER SHARED MEMORY: COMPUTES ONLY DOT PRODUCT ELEMENT HERE
		for (int iter2 = 0 ; iter2 < step ; iter2++)
			exy += a_shared[ty][iter2] * b_shared[iter2][tx];
		__syncthreads();
	}

	// MY IDEA: SHARE THE FIRST AND SECOND MOMENT INFORMATION BETWEEN DIFFERENT THREADS
	a_shared[ty][tx] = ex;
	b_shared[ty][tx] = ex2;
	ex = 0;
	ex2 = 0;
	__syncthreads();

	for (int iter = 0; iter < blockDim.x; iter++) {
		ex += a_shared[ty][iter];
		ex2 += b_shared[ty][iter];
	}

	__syncthreads();

	a_shared[ty][tx] = ey;
	b_shared[ty][tx] = ey2;
	ey = 0;
	ey2 = 0;

	__syncthreads();

	for (int iter = 0; iter < blockDim.y; iter++) {
		ey += a_shared[iter][tx];
		ey2 += b_shared[iter][tx];
	}

	__syncthreads();

	fexy = (float)((float)exy / (float)length);//20000.0;//length;
	fex = (float)((float)ex / (float)length);//20000.0;
	fex2 = (float)((float)ex2 / (float)length);//20000.0;
	fey = (float)((float)ey / (float)length);//20000.0;
	fey2 = (float)((float)ey2 / (float)length);//20000.0;

	float covXY = fexy - fex * fey;
	float varX  = fex2 - fex * fex;
	float varY  = fey2 - fey * fey;
	

	c_dev[row * width + col] = covXY / sqrt(varX * varY);
}




int main()
{
	int SZ = height * width;
	int *a = new int[height * length];
	int *b = new int[length * width];
	float *c = new float[height * width];

	// -------------------------
	// SET THE INPUT ARRAY
	int temp = 0;
	srand(time(NULL));
    
	for (int i = 0 ; i < height * length ; i++){
			temp = rand() % 100;
			a[i] = (int)temp;
	}

	for (int i = 0; i < length * width; i++) {
		temp = rand() % 100;
		b[i] = (int)temp;
	}

	// CORRELATION COEFICIENT ON DEVICE
	cudaError_t cudaStatus = computeCorrCoefOnDevice(c, a, b, SZ);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Correlation Coefficient on Device Failed!" << std::endl;
		return 1;
	}

	// CHECK THE DEVICE(GPU) WITH HOST(CPU)
	auto t1 = high_resolution_clock::now();
	checkResult(a, b, c);
	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1);

	std::cout << "Host exe time: " << duration.count() << std::endl;

    // FOR NVIDIA NSIGHT AND PROFILING. NOT NECESSARY IN THIS PROJECT
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	// FREEE ALL MEMORY ARRAYS
	delete a;
	delete b;
	delete c;

	system("pause");
    return 0;
}




cudaError_t computeCorrCoefOnDevice(float *c, const int *a, const int *b, int sz)
{
	int *a_dev = 0;
	int *b_dev = 0;
	float *c_dev = 0;
	cudaError_t cudaStatus;

	// CHOOSE WHICH GPU TO EXECUTE ON
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSEtDevice Failed! " << std::endl;
		goto Error;
	}


	// ---------------------------------
	// ALLOCATE MEMORY ON DEVICE
	cudaStatus = cudaMalloc((void**)&a_dev, height * length * sizeof(int));
	cudaStatus = cudaMalloc((void**)&b_dev, length * width * sizeof(int));
	cudaStatus = cudaMalloc((void**)&c_dev, height * width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc Failed!" << std::endl;
		goto Error;
	}

	// DATA TRANSFER FROM HOST TO DEVICE
	cudaStatus = cudaMemcpy(a_dev, a, height * length * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(b_dev, b, length * width * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMemcpy failed!" << std::endl;
		goto Error;
	}

	// --------------------------------
	// LAUNCH THE KERNEL
	dim3 blockSize(16, 32);
	int gridX = ceil((float)width / (float)blockSize.x);
	int gridY = ceil((float)height / (float)blockSize.y);
	dim3 gridSize(gridX, gridY);
	//dim3 gridSize(12, 8);
	

	auto t1 = high_resolution_clock::now();

	corrcoef_kernel_v1 << <gridSize, blockSize >> > (a_dev, b_dev, c_dev);

	// -------------------------------
	// SYNCRONIZE HOST AND DEVICE
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceSynchronize Failed!" << std::endl;
		goto Error;
	}

	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1);
	std::cout << "The kernel time: " << duration.count() << std::endl;


	// TRANSFER DATA FROM DEVICE TO HOST
	cudaStatus = cudaMemcpy(c, c_dev, height * width * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "ERROR: DATA TRANSFER FROM DEVICE TO HOST FAILED!" << std::endl;
		goto Error;
	}

	
Error:
	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);

	return cudaStatus;
}



void checkResult(int *a, int *b, float *c)
{
	int sum_b = 0, sum_a = 0.0, sum_b2 = 0.0, sum_a2 = 0.0, sum_ab = 0.0;
	int count = 0;
	float fsum_ab = 0.0, fsum_a = 0.0, fsum_a2 = 0.0, fsum_b = 0.0, fsum_b2 = 0.0;

	for (int iter = 0; iter < height; iter++)
		for (int iter2 = 0; iter2 < width; iter2++) {
			for (int iter3 = 0; iter3 < length; iter3++) {
				sum_a += a[iter*length + iter3];
				sum_b += b[iter3*width + iter2];
				sum_a2 += a[iter*length + iter3] * a[iter*length + iter3];
				sum_b2 += b[iter3*width + iter2] * b[iter3*width + iter2];
				sum_ab += a[iter*length + iter3] * b[iter3*width + iter2];
			}

/**/
			fsum_ab = (float)((float)sum_ab / (float)length); //(float)
			fsum_a = (float)((float)sum_a / (float)length); //(float)
			fsum_a2 = (float)((float)sum_a2 / (float)length); //(float)
			fsum_b = (float)((float)sum_b / (float)length); //(float)
			fsum_b2 = (float)((float)sum_b2 / (float)length); //(float)

			float cov = fsum_ab - fsum_a*fsum_b;
			float vara = fsum_a2 -fsum_a*fsum_a;
			float varb = fsum_b2 -fsum_b*fsum_b;

			float val_gpu = c[iter*width + iter2];
			float val_cpu = cov/sqrt(vara*varb);
			/**/
			if (val_cpu - val_gpu > 0.001 || val_gpu - val_cpu > 0.001) {
				count++;
				if (count < 10)
					std::cout << "DEV: " << val_gpu << ", HOST: " << val_cpu << std::endl;
			}
			/**/
			sum_a = 0.0;
			sum_a2 = 0.0;
			sum_b = 0.0;
			sum_b2 = 0.0;
			sum_ab = 0.0;
			/**/
		}
	std::cout << "ERROR RATE: " << count << std::endl;
}