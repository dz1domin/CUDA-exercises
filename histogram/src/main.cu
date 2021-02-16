#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

#include <cuda_runtime.h>

#define MAX_BINS 4096
#define THREADS_PER_BLOCK 512
#define SATURATION_VALUE 256

#define checkCudaErrors(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initInput (unsigned long int size, unsigned int* data, unsigned int binCount) {
	for(int i = 0; i < size; ++i) {
		data[i] = rand() % binCount;
	}
}

__global__ void cudaHistogramPrivate(unsigned long int dataSize, unsigned int* dataInput, unsigned int binCount, unsigned int* bins) {
	unsigned th = threadIdx.x;
	unsigned bx = blockDim.x;
	unsigned i = blockIdx.x * blockDim.x + th;
	unsigned gx = gridDim.x;
	unsigned stride = gx * bx;
	extern __shared__ unsigned int local_hist[];

	// init prywatnego histo
	for(unsigned int j = th; j < binCount; j += bx) {
		local_hist[j] = 0;
	}
	__syncthreads();

	// zliczanie
	for(unsigned int j = i; j < dataSize; j += stride) {
		atomicAdd(&(local_hist[dataInput[j]]), 1);
	}
	__syncthreads();

	// commit do vramu
	for(unsigned int j = th; j < binCount; j += bx) {
		atomicAdd(&(bins[j]), local_hist[j]);
	}
//	__syncthreads();
}

__global__ void cudaHistogramNonPrivate(unsigned long int dataSize, unsigned int* dataInput, unsigned int binCount, unsigned int* bins) {
	int th = threadIdx.x;
	int i = blockIdx.x * blockDim.x + th;

	// zliczanie
	for(unsigned int j = i; j < dataSize; j += blockDim.x * gridDim.x) {
		atomicAdd(&bins[dataInput[j]], 1);
	}
}

__global__ void cudaSaturateHistogram (unsigned int binCount, unsigned int* histogram) {
	int th = threadIdx.x;

	for(unsigned int j = th; j < binCount; j += blockDim.x) {
		if(histogram[j] > SATURATION_VALUE) {
			histogram[j] = SATURATION_VALUE;
		}
	}
}

void histogramGPU(unsigned long int dataSize, unsigned int* dataInput, int binCount, unsigned int* bins, int priv) {
	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));

	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));
	// odpalanie kerneli
	if(priv == 1) {
		cudaHistogramPrivate<<<120, THREADS_PER_BLOCK, binCount * sizeof(unsigned int)>>>(dataSize, dataInput, binCount, bins);
	}
	else {
		cudaHistogramNonPrivate<<<120, THREADS_PER_BLOCK>>>(dataSize, dataInput, binCount, bins);
	}

	cudaDeviceSynchronize();

	checkCudaErrors(cudaGetLastError());

	cudaSaturateHistogram<<<8, 512>>>(binCount, bins);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaEventRecord(stop, NULL));

	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	printf("GPU code run time: %f ms\n", msecTotal);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void histogramCPU(unsigned long int dataSize, unsigned int* dataInput, int binCount, unsigned int* bins) {
	timespec start, end;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	for(unsigned long int i = 0; i < dataSize; ++i) {
		if(bins[dataInput[i]] < SATURATION_VALUE) {
			bins[dataInput[i]]++;
		}
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	float msecTotal = (float)((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000;
	printf("CPU code run time: %f ms\n", msecTotal);
}

void checkBins(unsigned int binCount, unsigned int* cpuBins, unsigned int* gpuBins) {
	for(int i = 0; i < binCount; ++i) {
		if(cpuBins[i] != gpuBins[i]) {
			printf("BLAD!!!!!!! i = %d: %d != %d\n", i, cpuBins[i], gpuBins[i]);
			return;
		}
	}
}

int main(int argc, char** argv) {
	unsigned int priv, binCount;
	unsigned long int dataSize;
	char* endptr;

	if(argc != 4) {
		exit(-1);
	}

	dataSize = strtoul(argv[1], &endptr, 0);
	binCount = atoi(argv[2]);
	priv = atoi(argv[3]);

	if(binCount > MAX_BINS)
		binCount = MAX_BINS;

	cudaDeviceProp prop;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

	printf("running on: %s\ndata size: %lu bin count: %d private: %s\n", prop.name, dataSize, binCount, priv == 1 ? "true" : "false");

	srand(time(0));

	unsigned int* dataInput;
	checkCudaErrors(cudaMallocManaged(&dataInput, dataSize * sizeof(unsigned int)));

	initInput(dataSize, dataInput, binCount);

	unsigned int* bins;
	checkCudaErrors(cudaMallocManaged(&bins, binCount * sizeof(unsigned int)));

	for(unsigned int i = 0; i < binCount; ++i)
		bins[i] = 0;
	
	unsigned int cpuBins[binCount];

	for(unsigned int i = 0; i < binCount; ++i){
		cpuBins[i] = 0;
	}

	histogramCPU(dataSize, dataInput, binCount, cpuBins);

	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(unsigned int), 0));
	checkCudaErrors(cudaMemPrefetchAsync(bins, binCount * sizeof(unsigned int), 0));
	histogramGPU(dataSize, dataInput, binCount, bins, priv);

	checkBins(binCount, cpuBins, bins);

	checkCudaErrors(cudaFree(dataInput));
	checkCudaErrors(cudaFree(bins));

	printf("end\n");

	return EXIT_SUCCESS;
}
