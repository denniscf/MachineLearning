#include "FCM.h"
#include <chrono>
#include <string>
#include <vector>
#include <iostream>

void GenerateData(int nPoints, int nDims, std::vector<double> & data)
{
	unsigned int seed = 111;
	data.resize(nPoints*nDims);
	for (int i = 0; i < nPoints; i++)
		for (int j = 0; j < nDims; j++)
			data[i*nDims + j] = ((double)rand_r(&seed))/RAND_MAX;
}

void PrintMatrix(int nDim, const char* nameRow, std::vector<double> & matrix)
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

void PrintResults(int nDim, std::vector<double> & points, std::vector<double> & centroids, std::vector<double> & membership)
{
	int nCentroids = centroids.size()/nDim;
	int nPoints = points.size()/nDim;
	
	PrintMatrix(nDim, "Point", points);
	PrintMatrix(nDim, "Centroid", centroids);
	PrintMatrix(nPoints, "Membership", membership);
}

void TestClustering()
{
	int fuzziness = 2;
	int nPoints = 4;
	int nDims = 2;
	int nClusters = 2;

	std::vector<double> distances;
	std::vector<double> data;
	std::vector<double> centroids;
	GenerateData(nPoints, nDims, data);
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, 0.001, centroids);

	PrintResults(nDims, data, centroids, membership);
	printf("Done!\n");
}

void TestClusteringFixedVals()
{
	int fuzziness = 2;
	int nPoints = 10;
	int nDims = 2;
	int nClusters = 2;

	std::vector<double> distances;
	std::vector<double> data{0,0,0,1,1,0,5,5,5,6,6,7};
	std::vector<double> centroids;
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, 0.001, centroids);

	PrintResults(nDims, data, centroids, membership);
	printf("Done!");
}

void TestRunTime()
{
	//Seems to perform with similar precision when compared to fcm in matlab
	int fuzziness = 2;
	int nPoints = 1000000;
	int nDims = 10;
	int nClusters = 10;
	float precision = 0.01;
	
	std::vector<double> distances;
	std::vector<double> data;
	std::vector<double> centroids;
	
	GenerateData(nPoints, nDims, data);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, precision, centroids);
	auto t2 = std::chrono::high_resolution_clock::now();
	PrintMatrix(nDims, "Centroid", centroids);
	printf("Run time (milisecs): %li\n", std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1)/1000000); 
}

void TestGenerateInitMembership()
{
	int nPoints = 5;
	int nCentroids = 3;
	std::vector<double> membership;
	FCM::InitializeMembership(nPoints, nCentroids, membership);
	PrintMatrix(nPoints, "Cluster", membership);
}

int main(int argc, char** argv)
{
	//TestClusteringFixedVals();
	//TestClustering();
	//TestGenerateInitMembership();
	//TestClusteringFixedVals();
	TestRunTime();
	
	char ch;
	std::cin >> ch;
}
