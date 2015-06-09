all: algo 

algo: cudaFCM.o
	g++ main.cpp CUDAFCM.o -std=c++0x -o fcmCUDA -I/opt/cuda/6.0/include -L/opt/cuda/6.0/lib64 -lcuda -lcudart -lcurand

cudaFCM.o:	
	nvcc -arch=sm_20 -O3 -c CUDAFCM.cu -o CUDAFCM.o -lcuda -lcudart -lcurand 
clear:
	rm -rf *.o fcmCUDA 
