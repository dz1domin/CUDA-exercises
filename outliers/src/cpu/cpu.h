#pragma once

#include <stdio.h>
#include "../utility/utility.h"
#include <chrono>

template<typename inputType>
outlierResult findOutliersCPU(unsigned long int dataSize, inputType* dataInput, double sigmaCount, double sigma, double mean) {
	unsigned long int currentSize = 0;
	outlierResult res(dataSize);
	for(unsigned long int i = 0; i < dataSize; ++i) {
		if((dataInput[i] > mean + sigmaCount * sigma) || (dataInput[i] < mean - sigmaCount * sigma)) {
			res.getoutlierIndexArray()[currentSize] = i;
			currentSize++;
		}
	}
	res.setSize(currentSize);
	return res;
}

template<typename inputType>
outlierResult outliersCPU(unsigned long int dataSize, inputType* dataInput, double sigmaCount) {
	using namespace std::chrono;
	outlierResult res;
	double sigma, mean, variance;
	inputType sum = 0;
	inputType sq_sum = 0;
	auto start = high_resolution_clock::now();
	auto endStdDev = high_resolution_clock::now();

	if(dataSize != 0) {
		for(unsigned long int i = 0; i < dataSize; ++i) {
			sum += dataInput[i];
			sq_sum += dataInput[i] * dataInput[i];
		}

		mean = static_cast<double>(sum) / dataSize;
		variance = static_cast<double>(sq_sum) / dataSize - mean * mean;
		sigma = sqrt(variance);
		endStdDev = high_resolution_clock::now();

		res = findOutliersCPU<inputType>(dataSize, dataInput, sigmaCount, sigma, mean);
	}

	auto end = high_resolution_clock::now();

	printf("CPU std dev run time: %lf ms\nsigma: %lf\nsum: %lf\nmean: %lf\n", duration<double, std::milli>(endStdDev - start).count(), sigma, sum, mean);
	printf("CPU find run time: %lf ms\n", duration<double, std::milli>(end - endStdDev).count());
	printf("CPU code run time: %lf ms\n", duration<double, std::milli>(end - start).count());

	return res;
}
