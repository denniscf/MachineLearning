#include "FCM.h"
#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include "MathAux.h"

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

double TestRunTime(int nPoints, int nClusters)
{
	//Seems to perform with similar precision when compared to fcm in matlab
	int fuzziness = 2;
	int nDims = 3;
	float precision = 0.01;
	
	std::vector<double> distances;
	std::vector<double> data;
	std::vector<double> centroids;
	
	GenerateData(nPoints, nDims, data);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, precision, centroids);
	auto t2 = std::chrono::high_resolution_clock::now();
	return (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1).count()/1000000);
}

void TestGenerateInitMembership()
{
	int nPoints = 5;
	int nCentroids = 3;
	std::vector<double> membership;
	FCM::InitializeMembership(nPoints, nCentroids, membership);
	PrintMatrix(nPoints, "Cluster", membership);
}

void RunTimePointsIncrease()
{
	std::vector<int> nPoints {10000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000};
	std::vector<int> nCentroids {10};

	for(int i=0; i<nPoints.size();++i)
	{
		std::vector<double> time;
		for(int t=0; t<10;++t)
		{	
			time.push_back(TestRunTime(nPoints[i], nCentroids[0]));
		}
		double meanTime = mean(time);
		double stdevTime = stdev(time, meanTime);
		printf("nPoints: %d nClusters: %d Mean time (milisecs): %.3f Stdev time (milisecs): %.3f\n ",nPoints[i], nCentroids[0], meanTime, stdevTime); 
	}
	
}

void RunTimeClustersIncrease()
{
	std::vector<int> nPoints {1000000};
	std::vector<int> nCentroids {2, 3, 5, 7, 10, 13};
	for(int i=0; i<nPoints.size();++i)
		for(int j=0; j<nCentroids.size(); ++j)
			TestRunTime(nPoints[i], nCentroids[j]);
}

int main(int argc, char** argv)
{
	//TestClusteringFixedVals();
	//TestClustering();
	//TestGenerateInitMembership();
	//TestClusteringFixedVals();
	
	RunTimePointsIncrease();
	char ch;
	std::cin >> ch;
}
