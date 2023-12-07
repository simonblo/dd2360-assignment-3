#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define A (1<<25)
#define B 4096

typedef unsigned int uint32;

__global__ void gpuHistogram(uint32* bufferIn, uint32 bufferInSize, uint32* bufferOut, uint32 bufferOutSize)
{
	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < bufferInSize) atomicAdd(&bufferOut[bufferIn[tid]], 1);
}

__global__ void gpuHistogram2(uint32* bufferIn, uint32 bufferInSize, uint32* bufferOut, uint32 bufferOutSize)
{
	__shared__ uint32 groupshared[B];

	uint32 tid = threadIdx.x;
	uint32 gid = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < B; i += blockDim.x)
	{
		groupshared[i + tid] = 0;
	}

	if (gid < bufferInSize)
	{
		atomicAdd(&groupshared[bufferIn[gid]], 1);
	}

	for (int i = 0; i < B; i += blockDim.x)
	{
		atomicAdd(&bufferOut[i + tid], groupshared[i + tid]);
	}
}

__global__ void gpuSaturate(uint32* buffer, uint32 bufferSize, uint32 valueMin, uint32 valueMax)
{
	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < bufferSize) buffer[tid] = buffer[tid] > valueMax ? valueMax : buffer[tid];
}

int main()
{
	uint32* cpuBuffer1;
	uint32* cpuBuffer2;
	uint32* cpuBuffer3;
	uint32* gpuBuffer1;
	uint32* gpuBuffer2;

	srand(time(NULL));

	cpuBuffer1 = (uint32*)malloc(A * sizeof(uint32));
	cpuBuffer2 = (uint32*)malloc(B * sizeof(uint32));
	cpuBuffer3 = (uint32*)malloc(B * sizeof(uint32));

	//cudaHostAlloc((void**)&cpuBuffer1, A * sizeof(uint32), cudaHostAllocDefault);
	//cudaHostAlloc((void**)&cpuBuffer2, B * sizeof(uint32), cudaHostAllocDefault);
	//cudaHostAlloc((void**)&cpuBuffer3, B * sizeof(uint32), cudaHostAllocDefault);

	memset(cpuBuffer3, 0, B * sizeof(uint32));

	for (int i = 0; i != A; ++i)
	{
		uint32 v      = (uint32)rand() % (uint32)B;
		cpuBuffer1[i] = v;
		cpuBuffer3[v] = cpuBuffer3[v] + (cpuBuffer3[v] < 127);
	}

	cudaMalloc((void**)&gpuBuffer1, A * sizeof(uint32));
	cudaMalloc((void**)&gpuBuffer2, B * sizeof(uint32));

	uint32 errors  = 0;
	uint32 threads = 0;
	uint32 blocks  = 0;

	cudaMemcpy(gpuBuffer1, cpuBuffer1, A * sizeof(uint32), cudaMemcpyHostToDevice);
	cudaMemset(gpuBuffer2, 0, B * sizeof(uint32));

	threads = 64;
	blocks  = (A + threads - 1) / threads;
	gpuHistogram<<<blocks, threads>>>(gpuBuffer1, A, gpuBuffer2, B);

	threads = 64;
	blocks  = (B + threads - 1) / threads;
	gpuSaturate<<<blocks, threads>>>(gpuBuffer2, B, 0, 127);

	cudaDeviceSynchronize();
	cudaMemcpy(cpuBuffer2, gpuBuffer2, B * sizeof(uint32), cudaMemcpyDeviceToHost);

	for (int i = 0; i != B; ++i)
	{
		errors += (cpuBuffer2[i] != cpuBuffer3[i]);
	}

	printf("Elements: %u\n", A);
	printf("Errors:   %u\n", errors);

	cudaFree(gpuBuffer1);
	cudaFree(gpuBuffer2);

	free(cpuBuffer1);
	free(cpuBuffer2);
	free(cpuBuffer3);

	//cudaFreeHost(cpuBuffer1);
	//cudaFreeHost(cpuBuffer2);
	//cudaFreeHost(cpuBuffer3);

	return 0;
}