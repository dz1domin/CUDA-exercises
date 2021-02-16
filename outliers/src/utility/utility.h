#pragma once

#include <random>
#include <string>
#include <iostream>
#include <fstream>

#define checkCudaErrors(code) { cudaAssert((code), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename outputType>
inline void initInput (unsigned long int size, outputType* data, double mean, double stddev) {
	std::default_random_engine gen;
	std::normal_distribution<double> dist(mean, stddev);

	for(int i = 0; i < size; ++i) {
		data[i] = static_cast<outputType>(dist(gen));
//		data[i] = 1;
	}
}

class outlierResult {
private:
	unsigned long int size;
	unsigned long int capacity;
	unsigned long int* outlierIndexArray;
public:
	outlierResult() {
		this->capacity = 0;
		this->size = 0;
		this->outlierIndexArray = NULL;
	}

	outlierResult(unsigned long int inputSize) {
		this->capacity = inputSize;
		this->size = 0;
		if(inputSize == 0) {
			checkCudaErrors(cudaMallocManaged(&this->outlierIndexArray, 1 * sizeof(unsigned long int)));
		}
		else {
			checkCudaErrors(cudaMallocManaged(&this->outlierIndexArray, inputSize * sizeof(unsigned long int)));
		}
	}

	void free() {
		checkCudaErrors(cudaFree(this->outlierIndexArray));
	}

	void prefetchToDevice() {
		checkCudaErrors(cudaMemPrefetchAsync(this->outlierIndexArray, this->capacity * sizeof(unsigned long int), 0));
	}

	unsigned long int getSize() {
		return this->size;
	}

	unsigned long int* getoutlierIndexArray() {
		return this->outlierIndexArray;
	}

	void setoutlierIndexArray(unsigned long int* newArray) {
		this->outlierIndexArray = newArray;
	}

	void setSize(unsigned long int newSize) {
		this->size = newSize;
	}

	void dumpToFile(std::string filename) {
		std::ofstream file(filename);

		for(unsigned long int i = 0; i < this->size; i++) {
			file << this->outlierIndexArray[i] << std::endl;
		}

		file.close();
	}
};
