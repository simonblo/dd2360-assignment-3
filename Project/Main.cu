#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (1<<25)
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
	uint32* gpuBuffer1;
	uint32* gpuBuffer2;

	uint32 threads = 0;
	uint32 blocks  = 0;
	uint32 errors  = 0;

	srand(time(NULL));

	cudaHostAlloc((void**)&cpuBuffer1, N * sizeof(uint32), cudaHostAllocDefault);
	cudaHostAlloc((void**)&cpuBuffer2, B * sizeof(uint32), cudaHostAllocDefault);

	cudaMalloc((void**)&gpuBuffer1, N * sizeof(uint32));
	cudaMalloc((void**)&gpuBuffer2, B * sizeof(uint32));

	for (int i = 0; i != N; ++i)
	{
		cpuBuffer1[i] = (uint32)rand() % (uint32)B;
	}

	cudaMemcpy(gpuBuffer1, cpuBuffer1, N * sizeof(uint32), cudaMemcpyHostToDevice);
	cudaMemset(gpuBuffer2, 0, B * sizeof(uint32));

	threads = 1024;
	blocks  = (N + threads - 1) / threads;
	gpuHistogram<<<blocks, threads>>>(gpuBuffer1, N, gpuBuffer2, B);

	threads = 64;
	blocks  = (B + threads - 1) / threads;
	gpuSaturate<<<blocks, threads>>>(gpuBuffer2, B, 0, 127);

	cudaDeviceSynchronize();
	cudaMemcpy(cpuBuffer2, gpuBuffer2, B * sizeof(uint32), cudaMemcpyDeviceToHost);

	uint32 values[B];
	memset(values, 0, B * sizeof(uint32));

	for (int i = 0; i != N; ++i)
	{
		values[cpuBuffer1[i]] += values[cpuBuffer1[i]] < 127 ? 1 : 0;
	}

	for (int i = 0; i != B; ++i)
	{
		errors += (cpuBuffer2[i] != values[i]);
	}

	printf("Elements: %u\n", N);
	printf("Errors:   %u\n", errors);

	cudaFree(gpuBuffer1);
	cudaFree(gpuBuffer2);

	cudaFreeHost(cpuBuffer1);
	cudaFreeHost(cpuBuffer2);

	return 0;
}