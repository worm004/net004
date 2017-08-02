#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
#include "opencv2/opencv.hpp"
using namespace cv;
int main(){
	Net004 net;
	Parser parser0;
	parser0.read("../models/cifar_test.net004.net", "../models/cifar_test.net004.data", &net);

	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img0 = imread("../imgs/westerdam-ship-size.jpg");
	resize(img0,img0,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)img.data,0);

	DataLayer* l2 = (DataLayer*)ls["label"];
	l2->add_label(8,0);

	net.forward();
	if(abs(net.ls["loss"]->outputs[0].data[0] - 0.000024) < 1e-5)
		printf("[TEST] cifar10 test successful\n");
	else printf("[TEST] cifar10 test failed\n");
	return 0;
}
