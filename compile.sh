g++ -std=c++11 -o MTCNNFaceDetecion main.cpp mtcnnfacedetection.cpp -I. `pkg-config --cflags --libs opencv` -lglog -lboost_system -lcaffe
