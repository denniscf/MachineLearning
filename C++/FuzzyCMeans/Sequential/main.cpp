#include "FCM.h"
#include <chrono>
#include <string>
#include <vector>

void GenerateData(int nPoints, int nDims, std::vector<double> & data)
{
	data.resize(nPoints*nDims);
	for (int i = 0; i < nPoints; i++)
		for (int j = 0; j < nDims; j++)
			data[i*nDims + j] = ((double)rand())/RAND_MAX;
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
	
	PrintMatrix(nDim, "Point", points);
	PrintMatrix(nDim, "Centroid", centroids);
	PrintMatrix(nCentroids, "Membership", membership);
}

void TestClustering()
{
	int fuzziness = 3;
	int nPoints = 4;
	int nDims = 2;
	int nClusters = 2;

	std::vector<double> distances;
	std::vector<double> data;
	std::vector<double> centroids;
	GenerateData(nPoints, nDims, data);
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, 0.01, centroids);

	PrintResults(nDims, data, centroids, membership);
	printf("Done!");
}

void TestRunTime()
{
	int fuzziness = 3;
	int nPoints = 1000000;
	int nDims = 5;
	int nClusters = 10;

	std::vector<double> distances;
	std::vector<double> data;
	std::vector<double> centroids;

	GenerateData(nPoints, nDims, data);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, 0.01, centroids);
	auto t2 = std::chrono::high_resolution_clock::now();
	printf("Run time (milisecs): %li\n", std::chrono::duration_cast<std::chrono::nanoseconds>(t2 -t1)/1000000); 

}

int main(int argc, char** argv)
{
	TestRunTime();
}
