#include "FCM.h"
#include <string>
#include <vector>

void GenerateData(int nPoints, int nDims, std::vector<double> & data)
{
	data.resize(nPoints*nDims);
	for (int i = 0; i < nPoints; i++)
		for (int j = 0; j < nDims; j++)
			data[i*nDims + j] = rand();
}

void main()
{
	int fuzziness = 3;
	int nPoints = 100000;
	int nDims = 3;
	int nClusters = 10;
	
	std::vector<double> distances;
	std::vector<double> data;
	GenerateData(nPoints, nDims, data);

	std::vector<double> centroids;
	GenerateData(nPoints, nDims, data);
	std::vector<double> membership = FCM::RunClustering(nDims, data, nClusters, fuzziness, 0.01, centroids);
	printf("Done!");
}