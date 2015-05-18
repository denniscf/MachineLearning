#include "FCM.h"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdio.h>
#include <curand.h>
#include <cuda.h>
#include <vector>
/*
Code: Fuzzy C-means
Developer: Dennis Carnelossi Furlaneto
License: MIT
*/

namespace FCM
{
	#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	    printf("Error at %s:%d - ErroType:%s\n",__FILE__,__LINE__, cudaGetErrorString(x));\
	    return EXIT_FAILURE;}} while(0)
	#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    return EXIT_FAILURE;}} while(0)


	void PrintMatrix(int nDim, const char* nameRow, std::vector<float> & matrix)
	{
		int nRows = matrix.size()/nDim;
		
		for(int i=0; i<nRows; ++i)
		{
			printf("%s %d: ",nameRow, i);	
			for(int j=0; j<nDim; ++j)
			{
				printf("%.4f ", matrix[i*nDim + j]);	
			}
			printf("\n");
		}
	}

	/*RunClustering: This function receives a data vector with all the points with nDims dimensions and returns a membership vector with nPoints*nClusters positions. The membership values are in the following order: points -> membership per cluster.
			Inputs: nDims, data, nCentroids, fuzziness, errorThreshold
			Outputs: centroids
			Returns: membership
	*/
	std::vector<float> RunClustering(int nDims, const std::vector<float> & data, int nCentroids, float fuzziness, float errorThreshold, std::vector<float> & centroids)
	{
		std::vector<float> currentMembership;
		std::vector<float> nextMembership;
		int nPoints = data.size() / nDims;
		float* deviceNextMembership;
		std::vector<float> hostData;
		CudaInitializeMembership(nPoints, nCentroids, deviceNextMembership, nextMembership);

		float * deviceCentroids;
		float * deviceCurrentMembership;
		float * deviceDataPoints;
					
		cudaMalloc((void **)&deviceCentroids, nCentroids * nDims * sizeof(float));
		cudaMalloc((void **)&deviceCurrentMembership, nPoints * nCentroids * sizeof(float));
		cudaMalloc((void **)&deviceDataPoints, nPoints * nDims * sizeof(float));
		cudaMemcpy(deviceCentroids, &centroids[0], nCentroids * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(deviceDataPoints, &data[0], nPoints * nDims * sizeof(float), cudaMemcpyDeviceToHost);

/*		do
		{
			cudaMemcpyArrayToArray(deviceCurrentMembership, 0, 0, deviceNextMembership, 0, 0, 
				nPoints*nCentroids*sizeof(float), cudaMemcpyDeviceToDevice);
			FCM::ComputeCentroids(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, 
				deviceCurrentMembership, deviceCentroids);
			FCM::ComputeMembership(nDims, data, centroids, fuzziness, nextMembership);
		} while (IsMembershipDiffGreater(currentMembership, nextMembership, errorThreshold));
		FCM::ComputeCentroids(nCentroids, fuzziness, nDims, data, nextMembership, centroids);*/
		return nextMembership;
	}

	__global__ void NormalizePointsMembership(float* values, int nPoints, int nClusters)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
		if (idx < nPoints)	
		{
			int c;
			float sum=0.0;
			for( c = 0; c<nClusters; ++c)
				sum = sum + values[c*nPoints + idx];
			for( c = 0; c<nClusters; ++c)
				values[c*nPoints + idx] = values[c*nPoints + idx]/sum;
		}
	}
	
	int CudaInitializeMembership(int nPoints, int nClusters, float* deviceData, std::vector<float> & hostData)
	{    
		curandGenerator_t gen;

		/* Allocate n floats on host */
		int n = nPoints * nClusters;
		hostData.resize(n);
		
		/* Allocate n floats on device */
		CUDA_CALL(cudaMalloc((void **)&deviceData, n * sizeof(float)));

		/* Create pseudo-random number generator */
		CURAND_CALL(curandCreateGenerator(&gen,	CURAND_RNG_PSEUDO_DEFAULT));

		/* Set seed */
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
		
		/* Generate n floats on device */
		CURAND_CALL(curandGenerateUniform(gen, deviceData, n));
		
		int blockSize = n < 512 ? n : 512;
		int gridSize = ceil(float(n)/(blockSize*nClusters)) < 1 ? 1 : ceil(float(n)/(blockSize*nClusters));
		
		printf("BlockSize: %d\n", blockSize);
		printf("GridSize: %d\n", gridSize);
		NormalizePointsMembership<<<gridSize, blockSize>>>(deviceData, nPoints, nClusters);
		
		CUDA_CALL(cudaMemcpy(&hostData[0], deviceData, n * sizeof(float), cudaMemcpyDeviceToHost));

		/* Cleanup */
		CURAND_CALL(curandDestroyGenerator(gen));
		CUDA_CALL(cudaFree(deviceData));
		return EXIT_SUCCESS;
	}
	
	void TestComputeCentroid()
	{
		int fuzziness = 2;
		int nDims = 2;
		std::vector<float> data; 
		data.push_back(0.0); data.push_back(0.0);
		data.push_back(1.0); data.push_back(0.0);
		data.push_back(0.0); data.push_back(1.0);
		data.push_back(5.0); data.push_back(5.0);
		data.push_back(6.0); data.push_back(0.0);
		data.push_back(0.0); data.push_back(6.0);
		std::vector<float> centroids;
		centroids.push_back(0.33);centroids.push_back(0.33);
		centroids.push_back(5.3);centroids.push_back(5.3);
		std::vector<float> membership;
		membership.push_back(0.8);membership.push_back(0.8);membership.push_back(0.8);
		membership.push_back(0.2);membership.push_back(0.2);membership.push_back(0.2);
		membership.push_back(0.2);membership.push_back(0.2);membership.push_back(0.2);
		membership.push_back(0.8);membership.push_back(0.8);membership.push_back(0.8);
		int nPoints = data.size()/nDims;
		int nCentroids = centroids.size()/nDims;
		
		float * deviceCentroids;
		float * deviceCurrentMembership;
		float * deviceDataPoints;
					
		cudaMalloc((void **)&deviceCentroids, nCentroids * nDims * sizeof(float));
		cudaMalloc((void **)&deviceCurrentMembership, nPoints * nCentroids * sizeof(float));
		cudaMalloc((void **)&deviceDataPoints, nPoints * nDims * sizeof(float));
		cudaMemcpy(deviceCentroids, &centroids[0], nCentroids * nDims * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceDataPoints, &data[0], nPoints * nDims * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceCurrentMembership, &membership[0], nPoints * nCentroids * sizeof(float), cudaMemcpyHostToDevice);
		
		ComputeCentroids(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceCurrentMembership, deviceCentroids);
		printf("ComputeCentroids: Finished!\n");

	}
	/*InitializeMembership: Initializes randomly the membership value for each point for each of cluster.The sum of the membership values of a given point must be equal to 1. Membership nPointIdx*nClusters positions represent the memberships of the nPointIdx point.
		Inputs: nPoints, nClusters
		Outputs: membership
	*/
	void InitializeMembership(int nPoints, int nClusters, std::vector<float> & membership)
	{
		membership.resize(nPoints*nClusters);
		{
			unsigned int seed = (unsigned int)(time(NULL));
			for (int p = 0; p < nPoints; p++)
			{
				float sum = 0.0;
				int pIdx = p*nClusters;
				for (int c = 0; c < nClusters; c++)
				{
					membership[pIdx + c] = rand_r(&seed);
					sum += membership[pIdx + c];
				}
				for (int c = 0; c < nClusters; c++)
					membership[pIdx + c] /= sum;
			}
		}
	}

	/*IsMembershipDiffGreater: Check if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
		Inputs: currentMembership, nextMembership, error
		Outputs: True if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
	*/
	bool IsMembershipDiffGreater(const std::vector<float> & currentMembership, const std::vector<float> & nextMembership, float error)
	{
		int size = currentMembership.size();
		float calcError = 0.0;
		
		for (int u = 0; u < size; u++)
		{
			float diff = (nextMembership[u] - currentMembership[u]);
			calcError += diff*diff;
		}
		if(sqrt(calcError) > error)
			return true;
		return false;
	}


	__global__ void CudaComputeCentroids(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float* membership, float* centroids)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if (idx < nPoints)
		{
			__global__ float * numerator = (float*)malloc(nDim*sizeof(float));
			float denominator = 0.0;
			float weightedMembership;
			for (int c = 0; c < nCentroids; c++)
			{
				denominator = 0.0;
				weightedMembership = membership[c*nPoints + idx] * membership[c*nPoints + idx];
				//double weightedMembership = powf(membership[p*nCentroids + j], fuzziness);
				for (int d = 0; d < nDims; d++)
					numerator[d] += weightedMembership*data[idx*nDims + d];
				denominator += weightedMembership;
			}

			//Reduce values in the same block
			__syncthreads();
				
			//Reduce values for all grids
			__syncthreads();

			for (int c = 0; c < nCentroids; c++)
				for (int d = 0; d < nDims; d++)
					centroids[c*nDims + d] = numerator[d] / denominator;
			free(numerator);
		}	
	}


	
	void ComputeCentroids(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float * membership, float * centroids)
	{
		int blockSize = nPoints < 512 ? nPoints : 512;
		int gridSize = ceil(nPoints/blockSize) < 1 ? 1 : ceil(nPoints/blockSize);
		int numeratorSize = nDims;
		printf("ComputeCentroids > BlockSize = %d\n", blockSize);
		printf("ComputeCentroids > GridSize = %d\n", gridSize);
		
		CudaComputeCentroids<<<gridSize, blockSize, numeratorSize>>>(fuzziness, nPoints, nCentroids, nDims, data, membership, centroids);
	}

	__device__ float CudaComputeDistance(int nDims, int p1Idx, float * p1s,  int p2Idx, float * p2s)
	{
		float distance = 0;
		float pointDiff = 0;
		int p1IdxStart = p1Idx*nDims;
		int p2IdxStart = p2Idx*nDims;
		
		for (int d = 0; d < nDims; d++)
		{
			pointDiff = p1s[p1IdxStart + d] - p2s[p2IdxStart + d];
			distance += pointDiff * pointDiff;
		}
		return sqrt(distance);
	}

	__global__ void CudaComputeMembership(float fuzziness, int nDims, float * data, int nPoints, float * centroids, int nCentroids,  float * membership)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if (idx < nPoints)
		{
			for (int c1 = 0; c1 < nCentroids; ++c1)
			{
				float denominator = 0;
				float distPC = CudaComputeDistance(nDims, idx, data, c1, centroids);
				for (int c2 = 0; c2 < nCentroids; ++c2)
				{
					float divDistPC1DistPC2 = distPC / CudaComputeDistance(nDims, idx, data, c2, centroids);
					denominator += divDistPC1DistPC2 * divDistPC1DistPC2;
				}
				membership[idx + c1*nPoints] = 1./denominator;
			}
		}
		__syncthreads();
	
	}

	void ComputeMembershipWrapper(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float * membership, float * centroids)
	{
		int blockSize = nPoints < 512 ? nPoints : 512;
		int gridSize = ceil(nPoints/blockSize) < 1 ? 1 : ceil(nPoints/blockSize);
		int numeratorSize = nDims;
		
		CudaComputeMembership<<<gridSize, blockSize>>>(fuzziness, nDims, data, nPoints, centroids, nCentroids, membership);
	}


	void TestComputeMembership()
	{
		int fuzziness = 2;
		int nDims = 2;
		std::vector<float> data; 
		data.push_back(0.0); data.push_back(0.0);
		data.push_back(1.0); data.push_back(0.0);
		data.push_back(0.0); data.push_back(1.0);
		data.push_back(5.0); data.push_back(5.0);
		data.push_back(6.0); data.push_back(0.0);
		data.push_back(0.0); data.push_back(6.0);
		std::vector<float> centroids;
		centroids.push_back(0.33);centroids.push_back(0.33);
		centroids.push_back(5.3);centroids.push_back(5.3);
		std::vector<float> membership;
		membership.push_back(0.8);membership.push_back(0.8);membership.push_back(0.8);
		membership.push_back(0.2);membership.push_back(0.2);membership.push_back(0.2);
		membership.push_back(0.2);membership.push_back(0.2);membership.push_back(0.2);
		membership.push_back(0.8);membership.push_back(0.8);membership.push_back(0.8);
		int nPoints = data.size()/nDims;
		int nCentroids = centroids.size()/nDims;
		
		float * deviceCentroids;
		float * deviceCurrentMembership;
		float * deviceDataPoints;
					
		cudaMalloc((void **)&deviceCentroids, nCentroids * nDims * sizeof(float));
		cudaMalloc((void **)&deviceCurrentMembership, nPoints * nCentroids * sizeof(float));
		cudaMalloc((void **)&deviceDataPoints, nPoints * nDims * sizeof(float));
		cudaMemcpy(deviceCentroids, &centroids[0], nCentroids * nDims * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceDataPoints, &data[0], nPoints * nDims * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceCurrentMembership, &membership[0], nPoints * nCentroids * sizeof(float), cudaMemcpyHostToDevice);
		
		ComputeMembershipWrapper(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceCurrentMembership, deviceCentroids);
		
		cudaMemcpy(&centroids[0], deviceCentroids, nCentroids * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&data[0], deviceDataPoints, nPoints * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&membership[0], deviceCurrentMembership, nPoints * nCentroids * sizeof(float), cudaMemcpyDeviceToHost);
		
		PrintMatrix(nPoints, "Centroids", membership);

		printf("ComputeMembership: Finished!\n");

	}

}
