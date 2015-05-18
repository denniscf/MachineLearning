all: prog 

prog: cudaFCM.o
	g++ main.cpp FCM.o -std=c++0x -o prog -I/opt/cuda/6.0/include -L/opt/cuda/6.0/lib64 -lcuda -lcudart -lcurand

cudaFCM.o:	
	nvcc -arch=sm_20 -c FCM.cu -o FCM.o
clear:
	rm -rf *.o
