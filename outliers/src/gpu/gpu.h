#pragma once

#include <stdio.h>
#include "../utility/utility.h"
#include "../cpu/cpu.h"
#include <chrono>

#define THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 120

template<typename inputType>
__global__ void cudaReductionPartialStdDev(unsigned long int dataSize, inputType* dataInput, inputType* partialSums, inputType* partialSquaredSums) {
	__shared__ inputType localPartialSums[2 * THREADS_PER_BLOCK];
	__shared__ inputType localSquaredPartialSums[2 * THREADS_PER_BLOCK];
	unsigned int th = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * THREADS_PER_BLOCK;
	unsigned int i = start + 2 * th;

	// init lokalnej tablicy
	if(i < dataSize) {
		localPartialSums[2 * th] = dataInput[i];
		localSquaredPartialSums[2 * th] = dataInput[i] * dataInput[i];
	}
	else {
		localPartialSums[2 * th] = 0;
		localSquaredPartialSums[2 * th] = 0;
	}

	if(i + 1 < dataSize) {
		localPartialSums[2 * th + 1] = dataInput[i + 1];
		localSquaredPartialSums[2 * th + 1] = dataInput[i + 1] * dataInput[i + 1];
	}
	else {
		localPartialSums[2 * th + 1] = 0;
		localSquaredPartialSums[2 * th + 1] = 0;
	}

	// petla redukcji
	for(int stride = 1; stride <= THREADS_PER_BLOCK; stride *= 2) {
		__syncthreads();
		if(th % stride == 0) {
			localPartialSums[2 * th] += localPartialSums[2 * th + stride];
			localSquaredPartialSums[2 * th] += localSquaredPartialSums[2 * th + stride];
		}
	}

	// commit do vramu
	__syncthreads();
	partialSums[blockIdx.x] = localPartialSums[0];
	__syncthreads();
	partialSquaredSums[blockIdx.x] = localSquaredPartialSums[0];
}

template<typename inputType>
__global__ void cudaFindOutliers(unsigned long int dataSize, inputType* dataInput, double sigmaOffset, double mean, unsigned long int* outlierCounts, unsigned long int* outlierIndexesArray, unsigned int pointsPerThread) {
	unsigned long int th = threadIdx.x;
	unsigned long int i = blockIdx.x * blockDim.x + th;
	unsigned long int stride = blockDim.x * gridDim.x;
	unsigned long int globalStart = i * pointsPerThread;
	unsigned long int outlierCount = 0;

	for(unsigned long int j = i; j < dataSize; j += stride) {
		// if outlier found
//		__syncthreads();
		if(dataInput[j] > mean + sigmaOffset || dataInput[j] < mean - sigmaOffset) {
			outlierIndexesArray[globalStart + outlierCount] = j;

			outlierCount++;
		}
	}

	outlierCounts[i] = outlierCount;
}

template<typename inputType>
outlierResult outliersGPU(unsigned long int dataSize, inputType* dataInput, double sigmaCount) {
	using namespace std::chrono;
	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(inputType), 0));

	double sum = 0.0, sq_sum = 0.0;
	outlierResult res;
	inputType* partialSums;
	inputType* partialSquaredSums;
	unsigned int partialSumsSize = dataSize / (THREADS_PER_BLOCK * 2);
	partialSumsSize = dataSize % (THREADS_PER_BLOCK * 2) ? partialSumsSize + 1 : partialSumsSize;

	checkCudaErrors(cudaMallocManaged(&partialSums, partialSumsSize * sizeof(inputType)));
	checkCudaErrors(cudaMallocManaged(&partialSquaredSums, partialSumsSize * sizeof(inputType)));

	checkCudaErrors(cudaMemPrefetchAsync(partialSums, partialSumsSize * sizeof(inputType), 0));
	checkCudaErrors(cudaMemPrefetchAsync(partialSquaredSums, partialSumsSize * sizeof(inputType), 0));

	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));

	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	// odpalanie kerneli
	cudaReductionPartialStdDev<inputType><<<partialSumsSize, THREADS_PER_BLOCK>>>(dataSize, dataInput, partialSums, partialSquaredSums);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	checkCudaErrors(cudaMemPrefetchAsync(partialSums, partialSumsSize * sizeof(inputType), cudaCpuDeviceId));
	checkCudaErrors(cudaMemPrefetchAsync(partialSquaredSums, partialSumsSize * sizeof(inputType), cudaCpuDeviceId));

	// kod CPU - liczenie statystyk
	auto startCPU = high_resolution_clock::now();

	for(unsigned int j = 0; j < partialSumsSize; ++j) {
		sum += static_cast<double>(partialSums[j]);
		sq_sum += static_cast<double>(partialSquaredSums[j]);
	}
//	printf("sum: %lf\nsq_sum: %lf\n", sum, sq_sum);

	double mean = sum / dataSize;
	double variance = sq_sum / dataSize - mean * mean;
	double sigma = sqrt(variance);

	unsigned long int* outlierCounts;
	unsigned long int* outlierIndexesArray;
	unsigned int blocks = dataSize % THREADS_PER_BLOCK ? dataSize / THREADS_PER_BLOCK + 1 : dataSize / THREADS_PER_BLOCK;
	blocks = blocks > MAX_BLOCKS ? MAX_BLOCKS : blocks;
	unsigned int pointsPerThread = dataSize % (blocks * THREADS_PER_BLOCK) ? dataSize / (blocks * THREADS_PER_BLOCK) + 1 : dataSize / (blocks * THREADS_PER_BLOCK);
	unsigned long int outlierIndexesArraySize = blocks * THREADS_PER_BLOCK * pointsPerThread;
	unsigned long int countsSize = blocks * THREADS_PER_BLOCK;

	checkCudaErrors(cudaMallocManaged(&outlierCounts, countsSize * sizeof(unsigned long int)));
	checkCudaErrors(cudaMemPrefetchAsync(outlierCounts, countsSize * sizeof(unsigned long int), 0));

	checkCudaErrors(cudaMallocManaged(&outlierIndexesArray, outlierIndexesArraySize * sizeof(unsigned long int)));
	checkCudaErrors(cudaMemPrefetchAsync(outlierIndexesArray, outlierIndexesArraySize * sizeof(unsigned long int), 0));

	auto endCPU = high_resolution_clock::now();

	// drugi kernel co znajduje
	cudaEvent_t startFind;
	checkCudaErrors(cudaEventCreate(&startFind));
	cudaEvent_t stopFind;
	checkCudaErrors(cudaEventCreate(&stopFind));
	checkCudaErrors(cudaEventRecord(startFind, NULL));

	cudaFindOutliers<inputType><<<blocks, THREADS_PER_BLOCK>>>(dataSize, dataInput, sigmaCount * sigma, mean, outlierCounts, outlierIndexesArray, pointsPerThread);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stopFind, NULL));
	checkCudaErrors(cudaEventSynchronize(stopFind));
	float msecTotalGPUFind = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotalGPUFind, startFind, stopFind));
	checkCudaErrors(cudaMemPrefetchAsync(outlierCounts, countsSize * sizeof(inputType), cudaCpuDeviceId));
	checkCudaErrors(cudaMemPrefetchAsync(outlierIndexesArray, outlierIndexesArraySize * sizeof(inputType), cudaCpuDeviceId));

	auto startCPUFind = high_resolution_clock::now();

	unsigned int sumCounts = 0;
	for(int i = 0; i < countsSize; i++)
		sumCounts += outlierCounts[i];
//	printf("outlier count: %d\n", sumCounts);

	res = outlierResult(sumCounts);
	res.setSize(sumCounts);
//	printf("ppt: %u \n", pointsPerThread);
	unsigned long int index = 0;
	for(unsigned long int threadId = 0; threadId < countsSize; threadId++) {
		for(unsigned int outlierId = 0; outlierId < outlierCounts[threadId]; outlierId++) {
			res.getoutlierIndexArray()[index] = outlierIndexesArray[threadId * pointsPerThread + outlierId];
			index++;
		}
	}

	auto endCPUFind = high_resolution_clock::now();

	double msecCPUAFinddditional = duration<double, std::milli>(endCPUFind - startCPUFind).count();
	double msecCPUAdditional = duration<double, std::milli>(endCPU - startCPU).count();
	printf("GPU std dev run time: %lf ms\nsigma: %lf\nsum: %lf\nmean: %lf\n", msecCPUAdditional + msecTotal, sigma, sum, mean);
	printf("GPU find run time: %lf ms\n", msecTotalGPUFind + msecCPUAFinddditional);
	printf("GPU code run time: %lf ms\n", msecTotal + msecCPUAdditional + msecTotalGPUFind + msecCPUAFinddditional);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaEventDestroy(startFind));
	checkCudaErrors(cudaEventDestroy(stopFind));
	checkCudaErrors(cudaFree(partialSums));
	checkCudaErrors(cudaFree(partialSquaredSums));
	checkCudaErrors(cudaFree(outlierCounts));
	checkCudaErrors(cudaFree(outlierIndexesArray));

	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(inputType), cudaCpuDeviceId));

	return res;
}

template<typename inputType>
outlierResult outliersMixedGPU(unsigned long int dataSize, inputType* dataInput, double sigmaCount) {
	using namespace std::chrono;
	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(inputType), 0));

	inputType sum = 0, sq_sum = 0;
	outlierResult res;
	inputType* partialSums;
	inputType* partialSquaredSums;
	unsigned int partialSumsSize = dataSize / (THREADS_PER_BLOCK * 2);
	partialSumsSize = dataSize % (THREADS_PER_BLOCK * 2) ? partialSumsSize + 1 : partialSumsSize;

	checkCudaErrors(cudaMallocManaged(&partialSums, partialSumsSize * sizeof(inputType)));
	checkCudaErrors(cudaMallocManaged(&partialSquaredSums, partialSumsSize * sizeof(inputType)));

	checkCudaErrors(cudaMemPrefetchAsync(partialSums, partialSumsSize * sizeof(inputType), 0));
	checkCudaErrors(cudaMemPrefetchAsync(partialSquaredSums, partialSumsSize * sizeof(inputType), 0));

	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));
	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, NULL));

	// odpalanie kerneli
	cudaReductionPartialStdDev<inputType><<<partialSumsSize, THREADS_PER_BLOCK>>>(dataSize, dataInput, partialSums, partialSquaredSums);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	checkCudaErrors(cudaMemPrefetchAsync(partialSums, partialSumsSize * sizeof(inputType), cudaCpuDeviceId));
	checkCudaErrors(cudaMemPrefetchAsync(partialSquaredSums, partialSumsSize * sizeof(inputType), cudaCpuDeviceId));

	// kod CPU
	auto startCPUStdDev = high_resolution_clock::now();

	for(unsigned int j = 0; j < partialSumsSize; ++j) {
		sum += partialSums[j];
		sq_sum += partialSquaredSums[j];
	}
//	printf("sum: %lf\nsq_sum: %lf\n", sum, sq_sum);

	double mean = static_cast<double>(sum) / dataSize;
	double variance = static_cast<double>(sq_sum) / dataSize - mean * mean;
	double sigma = sqrt(variance);

	auto endCPUStdDev = high_resolution_clock::now();

	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(inputType), cudaCpuDeviceId));

	auto startCPU = high_resolution_clock::now();

	res = findOutliersCPU<inputType>(dataSize, dataInput, sigmaCount, sigma, mean);

	auto endCPU = high_resolution_clock::now();
	double msecTotalCPU = duration<double, std::milli>(endCPU - startCPU).count();
	double msecTotalCPUStdDev = duration<double, std::milli>(endCPUStdDev - startCPUStdDev).count();
	printf("GPU std dev run time: %lf ms\nsigma: %lf\nsum: %lf\nmean: %lf\n", msecTotalCPUStdDev + msecTotal, sigma, sum, mean);
	printf("CPU find run time: %lf ms\n", msecTotalCPU);
	printf("GPU/CPU code run time: %lf ms\n", msecTotal + msecTotalCPU + msecTotalCPUStdDev);

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	checkCudaErrors(cudaFree(partialSums));
	checkCudaErrors(cudaFree(partialSquaredSums));

	checkCudaErrors(cudaMemPrefetchAsync(dataInput, dataSize * sizeof(inputType), cudaCpuDeviceId));

	return res;
}
