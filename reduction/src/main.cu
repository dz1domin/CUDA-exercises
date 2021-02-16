#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024
#define DATA_SIZE 100000000

#define checkCudaErrors(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initInput (unsigned long int size, long int* data, long int initVal) {
	for(int i = 0; i < size; ++i) {
		data[i] = initVal;
	}
}

__global__ void cudaReductionPartialSums(unsigned long int dataSize, long int* dataInput, long int* partialSums) {
	__shared__ long int localData[2 * THREADS_PER_BLOCK];
	unsigned int th = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * THREADS_PER_BLOCK;
	unsigned int i = start + 2 * th;

	// init lokalnej tablicy
	if(i < dataSize) {
		localData[2 * th] = dataInput[i];
	}
	else {
		localData[2 * th] = 0;
	}

	if(i + 1 < dataSize) {
		localData[2 * th + 1] = dataInput[i + 1];
	}
	else {
		localData[2 * th + 1] = 0;
	}

	// petla redukcji
	for(int stride = 1; stride <= THREADS_PER_BLOCK; stride *= 2) {
		__syncthreads();
		if(th % stride == 0) {
			localData[2 * th] += localData[2 * th + stride];
		}
	}

	// commit do vramu
	partialSums[blockIdx.x] = localData[0];
}

long int outliersGPU(unsigned long int dataSize, long int* dataInput) {
	long int res = 0;
	long int* partialSums;
	unsigned int partialSumsSize = dataSize / (THREADS_PER_BLOCK * 2);
	partialSumsSize = dataSize % (THREADS_PER_BLOCK * 2) ? partialSumsSize + 1 : partialSumsSize;

	checkCudaErrors(cudaMallocManaged(&partialSums, partialSumsSize * sizeof(long int)));

	checkCudaErrors(cudaMemPrefetchAsync(partialSums, partialSumsSize * sizeof(long int), 0));

	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));

	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	// odpalanie kerneli
	cudaReductionPartialSums<<<partialSumsSize, THREADS_PER_BLOCK>>>(dataSize, dataInput, partialSums);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaEventRecord(stop, NULL));

	checkCudaErrors(cudaEventSynchronize(stop));

	for(unsigned int j = 0; j < partialSumsSize; ++j) {
		res += partialSums[j];
	}

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	printf("GPU code run time: %f ms\nresult: %d\n", msecTotal, res);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFree(partialSums));

	return res;
}

long int outliersCPU(unsigned long int dataSize, long int* dataInput) {
	timespec start, end;
	long int res = 0;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	for(unsigned long int i = 0; i < dataSize; ++i) {
		res += dataInput[i];
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	float msecTotal = (float)((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000;
	printf("CPU code run time: %f ms\nresult: %d\n", msecTotal, res);

	return res;
}

int main(int argc, char** argv) {
	cudaDeviceProp prop;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

	unsigned long int dataSize;
	char* endptr;

	if(argc != 2) {
		exit(-1);
	}

	dataSize = strtoul(argv[1], &endptr, 0);

	printf("running on: %s\ndata size: %lu\n", prop.name, dataSize);

	long int* dataInput;

	checkCudaErrors(cudaMallocManaged(&dataInput, dataSize * sizeof(long int)));

	initInput(dataSize, dataInput, 1);

	long int cpuRes = outliersCPU(dataSize, dataInput);

	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(long int), 0));

	long int gpuRes = outliersGPU(dataSize, dataInput);

	if(cpuRes != gpuRes) {
		printf("BLAD!!!!!!! %ld != %ld\n", cpuRes, gpuRes);
	}

	printf("end\n");

	checkCudaErrors(cudaFree(dataInput));

	return EXIT_SUCCESS;
}
