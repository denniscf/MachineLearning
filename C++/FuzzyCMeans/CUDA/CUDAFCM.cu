#include "CUDAFCM.h"
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

namespace CUDAFCM
{
	#define N_THREADS 256 
		
	#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	    printf("Error at %s:%d - ErroType:%s\n",__FILE__,__LINE__, cudaGetErrorString(x));\
	    return EXIT_FAILURE;}} while(0)
	#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	    printf("Error at %s:%d\n",__FILE__,__LINE__);\
	    return EXIT_FAILURE;}} while(0)

	#define CheckCudaErrors(){ \
		cudaError_t errorCode = cudaGetLastError();\
		if(errorCode != cudaSuccess)\
			printf("Check Cuda Error at %s, line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(errorCode));}
	
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

	
	__global__ void CudaCopyDataDeviceToDevice(volatile float * dst, const float * src, int nElems)
	{
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		if (idx < nElems)
			dst[idx] = src[idx];
	}
	
	void CopyDataDeviceToDevice(float * dst, const float * src, int nPoints)
	{
		int blockSize = nPoints < N_THREADS ? nPoints : N_THREADS;
		int gridSize = ceil(((float)nPoints)/blockSize) < 1 ? 1 : ceil(((float)nPoints)/blockSize);
		CudaCopyDataDeviceToDevice<<<gridSize, blockSize>>>(dst, src, nPoints);
		cudaDeviceSynchronize();
	}
	
	/*RunClustering: This function receives a data vector with all the points with nDims dimensions and returns a membership vector with nPoints*nClusters positions. The membership values are in the following order: points -> membership per cluster.
			Inputs: nDims, data, nCentroids, fuzziness, errorThreshold
			Outputs: centroids
			Returns: membership
	*/
	std::vector<float> RunClustering(int nDims, const std::vector<float> & data, float fuzziness, float errorThreshold, int nCentroids, std::vector<float> & centroids)
	{
		std::vector<float> currentMembership;
		std::vector<float> nextMembership;
		int nPoints = data.size() / nDims;
		int nMembershipSize = nPoints * nCentroids;
		cudaEvent_t start, stop;			
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
			
		float miliseconds;
		
		cudaEventRecord(start);	
		float* deviceNextMembership;
		CudaInitializeMembership(nPoints, nCentroids, &deviceNextMembership, nextMembership);
		cudaEventRecord(stop);	
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&miliseconds, start, stop);
		printf("CudaInitializeMembership: %f ms\n", miliseconds);
		CheckCudaErrors();

		float * deviceCentroids;
		float * deviceCurrentMembership;
		float * deviceDataPoints;
		float * deviceSquaredError;
		
		cudaMalloc((void**)&deviceCentroids, nCentroids * nDims * sizeof(float));
		cudaMemset(deviceCentroids, 0, nCentroids * nDims * sizeof(float));
		CheckCudaErrors();

		cudaMalloc((void**)&deviceCurrentMembership, nPoints * nCentroids * sizeof(float));
		cudaMalloc((void**)&deviceDataPoints, nPoints * nDims * sizeof(float));
		cudaMalloc((void**)&deviceSquaredError, sizeof(float));
		cudaMemcpy(deviceDataPoints, &data[0], nPoints * nDims * sizeof(float), cudaMemcpyHostToDevice);
		CheckCudaErrors();

		bool notConverged = !false;
		do
		{
			CopyDataDeviceToDevice( deviceCurrentMembership, deviceNextMembership, nPoints*nCentroids);
			//cudaMemcpy(deviceCurrentMembership, deviceNextMembership, nCentroids * nPoints * sizeof(float), cudaMemcpyDeviceToDevice);
			CheckCudaErrors();
			CUDAFCM::ComputeCentroids(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceCurrentMembership, deviceCentroids);
			CheckCudaErrors();
			CUDAFCM::ComputeMembership(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceNextMembership, deviceCentroids);
			CheckCudaErrors();
			notConverged = IsMembershipDiffGreater(deviceCurrentMembership, deviceNextMembership, nMembershipSize, errorThreshold, deviceSquaredError);
		} while(notConverged);
		CUDAFCM::ComputeCentroids(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceNextMembership, deviceCentroids);
		CheckCudaErrors();
	
		centroids.resize(nCentroids*nDims);	
		nextMembership.resize(nPoints*nCentroids);
		cudaMemcpy(&centroids[0], deviceCentroids, nCentroids * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&nextMembership[0], deviceCurrentMembership, nPoints * nCentroids * sizeof(float), cudaMemcpyDeviceToHost);
		CheckCudaErrors();
		
		cudaFree(deviceNextMembership);	
		cudaFree(deviceCurrentMembership);	
		cudaFree(deviceDataPoints);	
		cudaFree(deviceCentroids);	
		cudaFree(deviceSquaredError);
		CheckCudaErrors();
		
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
		__syncthreads();
	}
	
	int CudaInitializeMembership(int nPoints, int nClusters, float** deviceData, std::vector<float> & hostData)
	{    
		curandGenerator_t gen;

		int n = nPoints * nClusters;
		hostData.resize(n);
		
		cudaMalloc((void**)deviceData, n * sizeof(float));
		CheckCudaErrors();

		curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
		CheckCudaErrors();
		curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
		CheckCudaErrors();
		
		curandGenerateUniform(gen, *deviceData, n);
		CheckCudaErrors();
		
		int blockSize = n < N_THREADS ? n : N_THREADS;
		int gridSize = ceil(float(n)/(blockSize*nClusters)) < 1 ? 1 : ceil(float(n)/(blockSize*nClusters));
		NormalizePointsMembership<<<gridSize, blockSize>>>(*deviceData, nPoints, nClusters);
		CheckCudaErrors();
		
		cudaMemcpy(&hostData[0], *deviceData, n * sizeof(float), cudaMemcpyDeviceToHost);
		curandDestroyGenerator(gen);
		CheckCudaErrors();
		//CUDA_CALL(cudaFree(deviceData));
		return EXIT_SUCCESS;
	}

	__global__ void CudaIsMembershipDiffGreater(float * currentMembership, float * nextMembership, int size, float error, float * totalSquaredDiff)
	{
		int idx = threadIdx.x;
		int nThreads = blockDim.x;
		extern __shared__ float squaredDiffs[];
		squaredDiffs[idx] = 0.0;	
		
		int dataBlockSize;
		if(size < nThreads)
			dataBlockSize = 1;
		else
			dataBlockSize = ceil(((float)size) / nThreads); 

		int startPos = idx*dataBlockSize;
		int endPos = startPos + dataBlockSize;

		if (endPos >= size)
			endPos = size;

		if(startPos < size)
		{
			int i;
			float diff;
			float calcError = 0.0;
			
			for(i=startPos; i<endPos; ++i)
			{
				diff = (nextMembership[i] - currentMembership[i]);
				calcError += diff*diff;
			}
			squaredDiffs[idx] = calcError;
		}
		__syncthreads();
		
		float auxTotalSquareDiff = 0;		
		if (idx == 0)
		{
			for(int i = 0; i<nThreads;++i)
				auxTotalSquareDiff += squaredDiffs[i];
			totalSquaredDiff[0] = sqrt(auxTotalSquareDiff);
			printf("%f\n", totalSquaredDiff[0]);
		}
	}

	/*IsMembershipDiffGreater: Check if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
		Inputs: currentMembership, nextMembership, error
		Outputs: True if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
	*/
	bool IsMembershipDiffGreater(float * deviceCurrentMembership, float * deviceNextMembership, int size, float error, float * deviceSquaredError)
	{
		float miliseconds;
		cudaEvent_t start, stop;			
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		
		int blockSize = size < N_THREADS ? size : N_THREADS;

		CudaIsMembershipDiffGreater<<<1, blockSize, blockSize*sizeof(float)>>>(deviceCurrentMembership,deviceNextMembership, size, error, deviceSquaredError);
		CheckCudaErrors();
		
		float squaredError[1];
		cudaMemcpy(&squaredError[0], deviceSquaredError, sizeof(float), cudaMemcpyDeviceToHost);
		CheckCudaErrors();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&miliseconds, start, stop);
		printf("MembershipDiffGreater: %f ms\n", miliseconds);

		if(error >= squaredError[0])
			return false;
		return true;
	}

	void TestComputeSquareError()
	{
		std::vector<float> data1; 
		data1.push_back(0.5); data1.push_back(0.0);
		data1.push_back(1.0); data1.push_back(0.0);
		data1.push_back(0.0); data1.push_back(1.0);

		std::vector<float> data2; 
		data2.push_back(0.0); data2.push_back(0.0);
		data2.push_back(0.0); data2.push_back(0.0);
		data2.push_back(0.0); data2.push_back(0.0);

		int size = data1.size();
		
		float * dData1;
		float * dData2;
		float * deviceSquaredError;
		cudaMalloc((void **)&deviceSquaredError, sizeof(float));	
		cudaMalloc((void **)&dData1, size * sizeof(float));
		cudaMalloc((void **)&dData2, size * sizeof(float));
		cudaMemcpy(dData1, &data1[0], size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dData2, &data2[0], size * sizeof(float), cudaMemcpyHostToDevice);

		float error = 1;
		IsMembershipDiffGreater(dData1, dData2, size, error, deviceSquaredError);
		cudaFree(dData1);
		cudaFree(dData2);
		cudaFree(deviceSquaredError);
	}
		
	__global__ void CudaComputeCentroidsSingleBlock(float fuzziness, int nPoints, int idxCentroid, int nDims, float* data, float* membership, float* centroids)
	{
		int idx = threadIdx.x;
		int nThreads = blockDim.x;
		extern __shared__ float numerator[];
		__shared__ float denominator[N_THREADS];
			
		int dataBlockSize;
		if(nPoints < nThreads)
			dataBlockSize = 1;
		else
			dataBlockSize = ceil(((float)nPoints) / nThreads); 

		int startPos = idx*dataBlockSize;
		int endPos = startPos + dataBlockSize;

		if(startPos < nPoints && endPos > nPoints)
			endPos = nPoints;
		
		if(startPos < nPoints)
		{
			for(int i=0; i<nDims; ++i)
				numerator[idx*nDims + i] = 0.0;
			denominator[idx] = 0.0;

			register float weightedMembership;
			for (int p = startPos; p < endPos; ++p)
			{
				weightedMembership = powf(membership[idxCentroid*nPoints + p], 2);
				for (int d = 0; d < nDims; d++)
					numerator[idx*nDims + d] += weightedMembership*data[p*nDims + d];
				denominator[idx] += weightedMembership;
			}
		}
		
		__syncthreads();

		if (idx == 0)
		{	
			int i, j;
			for(i=1;i<nThreads;++i)
				denominator[0] += denominator[i];
			for(i=nDims; i<nThreads*nDims; i+=nDims)
				for(j=0; j<nDims;j++)
					numerator[j] += numerator[i+j];

			for (int d = 0; d < nDims; d++)
				centroids[idxCentroid*nDims + d] = numerator[d]/denominator[0];
		}
	}

	__global__ void CudaComputePartialCentroid(float fuzziness, int nPoints, int idxCentroid, int nDims, float* data, float* membership, float* centroids, int dataBlockSize, float *globalNumerator, float * globalDenominator)
	{
		int tGlobalIdx = blockDim.x*blockIdx.x + threadIdx.x;
		extern __shared__ float numerator[];
		__shared__ float denominator[N_THREADS];

		for(int i=0; i<nDims; ++i)
			numerator[threadIdx.x*nDims + i] = 0.0;
		denominator[threadIdx.x] = 0.0;
			
		if(tGlobalIdx == 0)
		{
			for(int i=0; i<nDims; ++i)
				globalNumerator[i] = 0.0;
			globalDenominator[0] = 0.0;
		}

		int startPos = tGlobalIdx*dataBlockSize;
		int endPos = startPos + dataBlockSize;

		if(startPos < nPoints && endPos > nPoints)
			endPos = nPoints;
		
		if(startPos < nPoints)
		{
			register float weightedMembership;
			for (int p = startPos; p < endPos; ++p)
			{
				weightedMembership = powf(membership[idxCentroid*nPoints + p], 2);
				for (int d = 0; d < nDims; d++)
					numerator[threadIdx.x*nDims + d] += weightedMembership*data[p*nDims + d];
				denominator[threadIdx.x] += weightedMembership;
			}
		}
		
		__syncthreads();
		if (threadIdx.x == 0)
		{
			for(int i=1;i<blockDim.x;++i)
				denominator[0] += denominator[i];
			for(int i=nDims; i<blockDim.x*nDims; ++i)
				numerator[i%nDims] += numerator[i];
		
			atomicAdd(&globalDenominator[0], (float)denominator[0]);
			for(int i=0; i<nDims;i++)
				atomicAdd(&globalNumerator[i], (float)numerator[i]);
		}
	}

	__global__ void CudaComputeOnlyCentroids(int idxCentroid, int nDims, float* centroids, float *globalNumerator, float * globalDenominator)
	{
		int tGlobalIdx = blockDim.x*blockIdx.x + threadIdx.x;
		if(tGlobalIdx == 0)
		{
			for (int d = 0; d < nDims; d++)
				centroids[idxCentroid*nDims + d] = globalNumerator[d]/globalDenominator[0];
		}
	}

	void ComputeCentroids(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float * membership, float * centroids)
	{
		float miliseconds;
		cudaEvent_t start, stop;			
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		int dataBlockSize, blockSize, gridSize;
		if(nPoints < N_THREADS)
		{
			gridSize = 1;
			dataBlockSize = 1;
			blockSize = nPoints;
		}
		else
		{	
			gridSize = 10;
			blockSize = N_THREADS;
			dataBlockSize = ceil((float)nPoints / (blockSize*gridSize)); 
		}
		int numeratorSize = blockSize*nDims;
		float * deviceNumerator;
		float * deviceDenominator;
		for(int i=0; i<nCentroids; ++i)
		{
			cudaMalloc((void **)&deviceNumerator, nDims * sizeof(float));
			cudaMalloc((void **)&deviceDenominator, sizeof(float));
			CudaComputePartialCentroid<<<gridSize, blockSize, numeratorSize*sizeof(float)>>>(fuzziness, nPoints, i, nDims, data, membership, centroids, 
				dataBlockSize, deviceNumerator, deviceDenominator);
			cudaDeviceSynchronize();
			CudaComputeOnlyCentroids<<<1, 1>>>(i, nDims, centroids, deviceNumerator, deviceDenominator);
			cudaFree(deviceNumerator);
			cudaFree(deviceDenominator);
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&miliseconds, start, stop);
		printf("ComputeCentroids: %f ms\n", miliseconds);
		CheckCudaErrors();
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
		data.push_back(5.0); data.push_back(6.0);
		data.push_back(6.0); data.push_back(7.0);
		std::vector<float> centroids;
		centroids.push_back(0);centroids.push_back(0);
		centroids.push_back(0);centroids.push_back(0);
		std::vector<float> membership;
		membership.push_back(0.9966);membership.push_back(0.9897);membership.push_back(0.99);
		membership.push_back(0.0250);membership.push_back(0.0020);membership.push_back(0.0185);
		membership.push_back(0.0034);membership.push_back(0.0103);membership.push_back(0.0100);
		membership.push_back(0.9750);membership.push_back(0.9980);membership.push_back(0.9815);
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
		
		cudaMemcpy(&centroids[0], deviceCentroids, nCentroids * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&data[0], deviceDataPoints, nPoints * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&membership[0], deviceCurrentMembership, nPoints * nCentroids * sizeof(float), cudaMemcpyDeviceToHost);
		
		PrintMatrix(nDims, "Points", data);
		PrintMatrix(nDims, "Centroids", centroids);

		cudaFree(deviceCurrentMembership);	
		cudaFree(deviceDataPoints);	
		cudaFree(deviceCentroids);	
		printf("ComputeCentroids: Finished!\n");

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
			register float denominator, distPC, divDistPC1DistPC2;
			extern __shared__ float sharedData[];
			int c1, c2;
			for(int i=0; i<nDims; ++i)
				sharedData[threadIdx.x*nDims + i] = data[idx*nDims+i];
			
			for (c1 = 0; c1 < nCentroids; ++c1)
			{
				denominator = 0.0;
				distPC = CudaComputeDistance(nDims, threadIdx.x, sharedData, c1, centroids);
				for (c2 = 0; c2 < nCentroids; ++c2)
				{
					divDistPC1DistPC2 = distPC / CudaComputeDistance(nDims, threadIdx.x, sharedData, c2, centroids);
					denominator += divDistPC1DistPC2 * divDistPC1DistPC2;
				}
				membership[idx + c1*nPoints] = 1./denominator;
			}
		}
		__syncthreads();
	
	}

	void ComputeMembership(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float * membership, float * centroids)
	{
		float miliseconds;
		cudaEvent_t start, stop;			
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		int blockSize = nPoints < N_THREADS ? nPoints : N_THREADS;
		int gridSize = ceil(((float)nPoints)/blockSize) < 1 ? 1 : ceil(((float)nPoints)/blockSize);
		CudaComputeMembership<<<gridSize, blockSize, blockSize*nDims*sizeof(float)>>>(fuzziness, nDims, data, nPoints, centroids, nCentroids, membership);
		cudaDeviceSynchronize();
		
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&miliseconds, start, stop);
		printf("ComputeMembership: %f ms\n", miliseconds);
		CheckCudaErrors();
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
		data.push_back(5.0); data.push_back(6.0);
		data.push_back(6.0); data.push_back(7.0);
		std::vector<float> centroids;
		centroids.push_back(0.3334);centroids.push_back(0.3337);
		centroids.push_back(5.3307);centroids.push_back(6.0040);
		std::vector<float> membership;
		membership.push_back(0);membership.push_back(0);membership.push_back(0);
		membership.push_back(0);membership.push_back(0);membership.push_back(0);
		membership.push_back(0);membership.push_back(0);membership.push_back(0);
		membership.push_back(0);membership.push_back(0);membership.push_back(0);
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
		
		ComputeMembership(fuzziness, nPoints, nCentroids, nDims, deviceDataPoints, deviceCurrentMembership, deviceCentroids);
		
		cudaMemcpy(&centroids[0], deviceCentroids, nCentroids * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&data[0], deviceDataPoints, nPoints * nDims * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&membership[0], deviceCurrentMembership, nPoints * nCentroids * sizeof(float), cudaMemcpyDeviceToHost);
	
		PrintMatrix(nDims, "Point", data);	
		PrintMatrix(nDims, "Centroid", centroids);	
		PrintMatrix(nPoints, "Membership", membership);

		cudaFree(deviceCurrentMembership);	
		cudaFree(deviceDataPoints);	
		cudaFree(deviceCentroids);	
		printf("ComputeMembership: Finished!\n");

	}

}
