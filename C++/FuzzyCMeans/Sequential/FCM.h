/* 
	Code: Fuzzy C-means
	Developer: Dennis Carnelossi Furlaneto    E-mail: dennis.furlaneto@gmail.com 
	License: MIT
*/

#include <vector>
//This implementation is based on the nice explanation of the algorithm given on http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/cmeans.html

namespace FCM
{
	std::vector<double> RunClustering(int nDims, const std::vector<double> & data, int nCentroids, double fuzziness, double errorThreshold, std::vector<double> & centroids);
	void InitializeMembership(int nPoints, int nClusters, std::vector<double> & membership);
	bool IsMembershipDiffGreater(const std::vector<double> & currentMembership, const std::vector<double> & nextMembership, double error);
	void ComputeCentroids(int nCentroids, double fuzziness, int nDims, const std::vector<double> & data, const std::vector<double> & membership, std::vector<double> & centroids);
	void ComputeMembership(int nDims, const std::vector<double> & data, const std::vector<double> & centroids, double fuzziness, std::vector<double> & membership);
	inline float ComputeDistance(int nDims, int p1Idx, const std::vector<double> & p1s, int p2Idx, const std::vector<double> & p2s);
};

