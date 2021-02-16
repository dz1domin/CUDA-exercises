#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>
#include "utility/utility.h"
#include "cpu/cpu.h"
#include "gpu/gpu.h"

#define NORMAL_MEAN 0.0
#define NORMAL_STDDEV 3.0

int cmp(const void *a, const void *b) {
	return *(unsigned long int*)a > *(unsigned long int*)b;
}

void validateResults(outlierResult& cpuRes, outlierResult& gpucpuRes, outlierResult& gpuRes) {
	qsort(gpuRes.getoutlierIndexArray(), gpuRes.getSize(), sizeof(unsigned long int), cmp);

	for(unsigned long int i = 0; i < cpuRes.getSize(); i++) {
		if(cpuRes.getoutlierIndexArray()[i] != gpucpuRes.getoutlierIndexArray()[i]) {
			printf("GPU/CPU - [%lu] %lu != %lu\n", i, cpuRes.getoutlierIndexArray()[i], gpucpuRes.getoutlierIndexArray()[i]);
		}

		if(cpuRes.getoutlierIndexArray()[i] != gpuRes.getoutlierIndexArray()[i]) {
			printf("GPU - [%lu] %lu != %lu\n", i, cpuRes.getoutlierIndexArray()[i], gpuRes.getoutlierIndexArray()[i]);
		}
	}
}

template<typename inputType>
void dumpInputData(unsigned long int dataSize, inputType* dataInput) {
	std::ofstream file("inputData.dat");

	for(unsigned long int i = 0; i < dataSize; i++) {
		file << dataInput[i] << std::endl;
	}

	file.close();
}

int main(int argc, char** argv) {
	typedef double testingType;
	cudaDeviceProp prop;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));

	unsigned long int dataSize;
	double mean, stddev, sigmas;

	if(argc != 5) {
		exit(-1);
	}

	dataSize = std::stoul(argv[1], nullptr, 0);
	mean = std::stod(argv[2]);
	stddev = std::stoul(argv[3]);
	sigmas = std::stoul(argv[4]);

	printf("running on: %s\ndata size: %lu\nsigmas count: %lf\ninitializing normal distribution with\nmean: %lf\nstddev: %lf\n", prop.name, dataSize, sigmas, mean, stddev);

	testingType* dataInput;

	checkCudaErrors(cudaMallocManaged(&dataInput, dataSize * sizeof(testingType)));

	initInput<testingType>(dataSize, dataInput, mean, stddev);

	printf("-----------------CPU APPROACH-----------------\n");

	auto cpuRes = outliersCPU<testingType>(dataSize, dataInput, sigmas);


	printf("-----------------MIX APPROACH-----------------\n");

	auto gpucpuRes = outliersMixedGPU<testingType>(dataSize, dataInput, sigmas);

	printf("-----------------GPU APPROACH-----------------\n");

	auto gpuRes = outliersGPU<testingType>(dataSize, dataInput, sigmas);

	printf("-----------------OUTLIER STAT-----------------\n");

	printf("CPU res size: %ld\nGPU/CPU res size: %ld\nGPU res size: %ld\n", cpuRes.getSize(), gpucpuRes.getSize(), gpuRes.getSize());

//	cpuRes.dumpToFile("cpuRes.dat");
//	gpucpuRes.dumpToFile("gpucpuRes.dat");
//	gpuRes.dumpToFile("gpuRes.dat");
//	dumpInputData<testingType>(dataSize, dataInput);

	validateResults(cpuRes, gpucpuRes, gpuRes);

	printf("end\n");

	checkCudaErrors(cudaFree(dataInput));
	cpuRes.free();
	gpucpuRes.free();
	gpuRes.free();

	return EXIT_SUCCESS;
}
