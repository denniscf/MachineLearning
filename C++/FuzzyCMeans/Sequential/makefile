OMP:
	g++ -O3 -fopenmp -DUSE_OMP -std=c++0x main.cpp FCM.cpp -o fcmOMP
SEQ:
	g++ -O3 -std=c++0x main.cpp FCM.cpp -o fcmSeq

clear:
	rm -rf fcmSeq fcmOmp
