#include <vector>
#include <cmath>

template<typename T>
double mean ( std::vector<T> & v )
{
        double sum = 0.0;
        int size = v.size();
       
        for ( int i=0; i < size; i++)
        {
            sum += v[i];
        }
       
        return sum / size;

};

template<typename T>
double stdev ( std::vector<T> & v , double mean )
{
        double sum = 0.0;
	int size = v.size();
        for ( int i =0; i < size; i++)
        {
            sum = (v[i] - mean)*(v[i] - mean);
        }
       
        return sqrt(sum/size);
};
