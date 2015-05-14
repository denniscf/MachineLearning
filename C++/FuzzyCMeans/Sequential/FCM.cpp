#include "FCM.h"
#include <iostream>
#include <cmath>
#include <omp.h>
/*
Code: Fuzzy C-means
Developer: Dennis Carnelossi Furlaneto
License: MIT
*/

namespace FCM
{
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

	/*RunClustering: This function receives a data vector with all the points with nDims dimensions and returns a membership vector with nPoints*nClusters positions. The membership values are in the following order: points -> membership per cluster.
			Inputs: nDims, data, nCentroids, fuzziness, errorThreshold
			Outputs: centroids
			Returns: membership
	*/
	std::vector<double> RunClustering(int nDims, const std::vector<double> & data, int nCentroids, double fuzziness, double errorThreshold, std::vector<double> & centroids)
	{
		std::vector<double> currentMembership;
		std::vector<double> nextMembership;
		int nPoints = data.size() / nDims;
		InitializeMembership(nPoints, nCentroids, nextMembership);

		do
		{
			currentMembership = nextMembership;
			FCM::ComputeCentroids(nCentroids, fuzziness, nDims, data, currentMembership, centroids);
			FCM::ComputeMembership(nDims, data, centroids, fuzziness, nextMembership);
		} while (IsMembershipDiffGreater(currentMembership, nextMembership, errorThreshold));
		
		FCM::ComputeCentroids(nCentroids, fuzziness, nDims, data, nextMembership, centroids);
		return nextMembership;
	}

	/*InitializeMembership: Initializes randomly the membership value for each point for each of cluster.The sum of the membership values of a given point must be equal to 1. Membership nPointIdx*nClusters positions represent the memberships of the nPointIdx point.
		Inputs: nPoints, nClusters
		Outputs: membership
	*/
	void InitializeMembership(int nPoints, int nClusters, std::vector<double> & membership)
	{
		membership.resize(nPoints*nClusters);
		#pragma omp parallel shared(nClusters, membership, nPoints) default(none) 	
		{
			unsigned int seed = (unsigned int)(time(NULL)) * omp_get_thread_num();
			#pragma omp parallel for 
			for (int p = 0; p < nPoints; p++)
			{
				double sum = 0.0;
				for (int c = 0; c < nClusters; c++)
				{
					membership[p + c*nPoints] = rand_r(&seed);
					sum += membership[p + c*nPoints];
				}
				for (int c = 0; c < nClusters; c++)
					membership[p + c*nPoints] /= sum;
			}
		}
	}

	/*IsMembershipDiffGreater: Check if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
		Inputs: currentMembership, nextMembership, error
		Outputs: True if the absolute difference between nextMembership[u] and currentMembership[u] are larger than a given error
	*/
	bool IsMembershipDiffGreater(const std::vector<double> & currentMembership, const std::vector<double> & nextMembership, double error)
	{
		int size = currentMembership.size();
		double calcError = 0.0;
		
		#pragma omp parallel for reduction(+:calcError) 
		for (int u = 0; u < size; u++)
		{
			double diff = (nextMembership[u] - currentMembership[u]);
			calcError += diff*diff;
		}
		if(sqrt(calcError) > error)
			return true;
		return false;
	}

	/*ComputeCentroids: Given a list of points, their nDims dimensions, the number of centroids that you want and the membership of each point to each cluster, calculate the next centroids
		Inputs: nCentroids, fuzziness, nDims, data, membership
		Outputs: centroids
	*/
	void ComputeCentroids(int nCentroids, double fuzziness, int nDims, const std::vector<double> & data, const std::vector<double> & membership, std::vector<double> & centroids)
	{
		if (centroids.size() == 0)
			centroids.resize(nCentroids*nDims);

		int nPoints = data.size() / nDims;

		#pragma omp parallel for
		for (int c = 0; c < nCentroids; c++)
		{
			std::vector<double> numerator(nDims);
			double denominator = 0.0;

			for (int p = 0; p < nPoints; p++)
			{
				double weightedMembership = membership[c*nPoints + p] * membership[c*nPoints + p];
				//double weightedMembership = powf(membership[p*nCentroids + j], fuzziness);
				for (int d = 0; d < nDims; d++)
					numerator[d] += weightedMembership*data[p*nDims + d];
				denominator += weightedMembership;
			}
			for (int d = 0; d < nDims; d++)
				centroids[c*nDims + d] = numerator[d] / denominator;
		}
	}

	/*ComputeMembership: Given a list of points, their nDims dimensions, the number of centroids that you want and the membership of each point to each cluster, calculate the membership of
		each point to each cluster. The membership is ordered by points and their membership.
		Inputs: nCentroids, nDims, data, centroids, fuzziness
		Outputs: membership
	*/
	void ComputeMembership(int nDims, const std::vector<double> & data, const std::vector<double> & centroids, double fuzziness, std::vector<double> & membership)
	{
		int nPoints = data.size() / nDims;
		int nCentroids = centroids.size() / nDims;
		//Exponent is not in use. powf is too slow
		double exponent = 2. / (fuzziness - 1);
		#pragma omp parallel for
		for (int j = 0; j < nPoints; ++j)
		{
			for (int i = 0; i < nCentroids; ++i)
			{
				float denominator = 0;
				float distPCij = ComputeDistance(nDims, j, data, i, centroids);
				for (int k = 0; k < nCentroids; ++k)
				{
					float divDistPCijDistPCik = distPCij / ComputeDistance(nDims, j, data, k, centroids);
					denominator += divDistPCijDistPCik * divDistPCijDistPCik;
					//denominator += powf(distPCij / ComputeDistance(nDims, i, data, k, centroids), exponent);
				}
				membership[j + i*nPoints] = 1./denominator;
			}
		}
	}

	/*ComputeDistance: 
		Inputs: 
			nDims: Number of dimensions of each point
			p1Idx: Index of point inside p1s
			p1s: List of points p1
			p2Idx:  Index of point inside p2s
			p2s: List of points p2
		Returns: distances between p1[p1Idx...p1Idx + nDims] & p2[p2Idx...p2Idx + nDims]
	*/
	inline float ComputeDistance(int nDims, int p1Idx, const std::vector<double> & p1s, int p2Idx, const std::vector<double> & p2s)
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
		return sqrtf(distance);
	}

}
