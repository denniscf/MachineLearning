/* 
	Code: Fuzzy C-means
	Developer: Dennis Carnelossi Furlaneto    E-mail: dennis.furlaneto@gmail.com 
	License: MIT
*/

#include <vector>
//This implementation is based on the nice explanation of the algorithm given on http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html

namespace FCM
{
	void TestComputeCentroid();
	int CudaInitializeMembership(int nPoints, int nClusters, float* deviceData, std::vector<float> & hostData);
	std::vector<float> RunClustering(int nDims, const std::vector<float> & data, int nCentroids, float fuzziness, float errorThreshold, std::vector<float> & centroids);
	void InitializeMembership(int nPoints, int nClusters, std::vector<float> & membership);
	bool IsMembershipDiffGreater(const std::vector<float> & currentMembership, const std::vector<float> & nextMembership, float error);
	void ComputeCentroids(float fuzziness, int nPoints, int nCentroids,  int nDims, float * data, float * membership, float * centroids);
	void ComputeMembership(int nDims, const std::vector<float> & data, const std::vector<float> & centroids, float fuzziness, std::vector<float> & membership);
	inline float ComputeDistance(int nDims, int p1Idx, const std::vector<float> & p1s, int p2Idx, const std::vector<float> & p2s);
	void TestComputeMembership();
};

