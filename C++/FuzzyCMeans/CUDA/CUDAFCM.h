/* 
	Code: Fuzzy C-means
	Developer: Dennis Carnelossi Furlaneto    E-mail: dennis.furlaneto@gmail.com 
	License: MIT
*/

#include <vector>
//This implementation is based on the nice explanation of the algorithm given on http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html

namespace CUDAFCM
{
	int CudaInitializeMembership(int nPoints, int nClusters, float** deviceData, std::vector<float> & hostData);
	std::vector<float> RunClustering(int nDims, const std::vector<float> & data, float fuzziness, float errorThreshold, int nCentroids, std::vector<float> & centroids);
	bool IsMembershipDiffGreater(float * currentMembership, float * nextMembership, int size, float error, float * deviceSquaredError);
	void ComputeCentroidMembership(float fuzziness, int nPoints, int nCentroids,  int nDims, float * data, float * membership, float * centroids);
	void ComputeCentroids(float fuzziness, int nPoints, int nCentroids,  int nDims, float * data, float * membership, float * centroids );
	void ComputeMembership(float fuzziness, int nPoints, int nCentroids, int nDims, float* data, float * membership, float * centroids);
	inline float ComputeDistance(int nDims, int p1Idx, const std::vector<float> & p1s, int p2Idx, const std::vector<float> & p2s);

	void TestComputeMembership();
	void TestComputeCentroid();
	void TestComputeSquareError();
};

