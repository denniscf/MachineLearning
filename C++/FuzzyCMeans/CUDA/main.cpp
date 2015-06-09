#include "CUDAFCM.h"
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <cuda.h>

void GenerateData(int nPoints, int nDims, std::vector<float> & data)
{
	unsigned int seed = 111;
	data.resize(nPoints*nDims);
	for (int i = 0; i < nPoints; i++)
		for (int j = 0; j < nDims; j++)
			data[i*nDims + j] = ((float)rand_r(&seed))/RAND_MAX;
}

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

void PrintResults(int nDim, std::vector<float> & points, std::vector<float> & centroids, std::vector<float> & membership)
{
	int nCentroids = centroids.size()/nDim;
	
	PrintMatrix(nDim, "Point", points);
	PrintMatrix(nDim, "Centroid", centroids);
	PrintMatrix(points.size()/nDim, "Membership", membership);
}

void TestClustering()
{
	int fuzziness = 2;
	int nPoints = 4;
	int nDims = 2;
	int nCentroids = 2;

	std::vector<float> distances;
	std::vector<float> data;
	std::vector<float> centroids;
	GenerateData(nPoints, nDims, data);
	std::vector<float> membership = CUDAFCM::RunClustering(nDims, data, fuzziness, 0.001, nCentroids, centroids);

	PrintResults(nDims, data, centroids, membership);
	printf("Done!\n");
}

void TestCUDARandomGenerator()
{
	int nPoints = 10;
	int nClusters = 5;
	float * deviceData;
	std::vector<float> hostData;
	CUDAFCM::CudaInitializeMembership(nPoints, nClusters, &deviceData, hostData);
	float sum=0.0;
	
	for(int i=0; i<nPoints; i++)
	{
		for (int j=0; j<nClusters; ++j)
			sum += hostData[j*nPoints + i];
		printf("point membership sum: %f\n", sum);
		sum =0.0;
	}	
		
	printf("TestCUDARandomGenerator finished!\n");
}

void TestRunTime()
{
	//Seems to perform with similar precision when compared to fcm in matlab
	int fuzziness = 2;
	int nPoints = 1000000;
	int nDims = 10;
	int nCentroids= 10;
	float precision = 0.001;
	
	std::vector<float> distances;
	std::vector<float> data;
	std::vector<float> centroids;
	
	GenerateData(nPoints, nDims, data);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::vector<float> membership = CUDAFCM::RunClustering(nDims, data, fuzziness, precision, nCentroids, centroids);
	auto t2 = std::chrono::high_resolution_clock::now();
	PrintMatrix(nDims, "Centroid", centroids);
	printf("Run time (milisecs): %li\n", std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1)/1000000); 
}

int main(int argc, char** argv)
{
	/*TestCUDARandomGenerator();
	CUDAFCM::TestComputeCentroid();
	CUDAFCM::TestComputeMembership();
	CUDAFCM::TestComputeSquareError();
	TestClustering();*/
	TestRunTime();
	
	char ch;
	std::cin >> ch;
}
