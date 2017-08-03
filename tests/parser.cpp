#include <string>
#include "opencv2/opencv.hpp"
#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
using namespace cv;
using namespace std;

void test_cifar10(){
	string img_path = "../imgs/westerdam-ship-size.jpg";
	int label = 8;
	Mat img = imread(img_path);

	Parser parser;
	string net_path0 = "../models/cifar.net004.net",
	       model_path0 = "../models/cifar.net004.data",
	       net_path1 = "cifar.net004.net",
	       model_path1 = "cifar.net004.data";

	Net004 net0;
	parser.read(net_path0, model_path0, &net0);
	parser.write(&net0,net_path1,model_path1);
	Layers & ls0 = net0.ls;
	DataLayer* l0 = (DataLayer*)ls0["data"];

	resize(img,img,Size(l0->outputs[0].h, l0->outputs[0].w));
	l0->add_image((uchar*)img.data,0,127,127,127);
	((DataLayer*)ls0["label"])->add_label(label,0);
	net0.forward();
	float score0 = net0.ls["loss"]->outputs[0].data[0];

	Net004 net1;
	parser.read(net_path1, model_path1, &net1);
	Layers & ls1 = net1.ls;
	DataLayer* l1 = (DataLayer*)ls1["data"];
	l1->add_image((uchar*)img.data,0,127,127,127);
	((DataLayer*)ls1["label"])->add_label(label,0);
	net1.forward();
	float score1 = net1.ls["loss"]->outputs[0].data[0];

	bool ret = abs(score0 - score1) < 1e-5;
	printf("[TEST] parser read/write cifar10 %s\n",ret?"sucessful":"failed");
}
void test_vgg10(){
	string img_path = "../imgs/westerdam-ship-size.jpg";
	int label = 628;
	Mat img = imread(img_path);

	Parser parser;

	string net_path0 = "../models/vgg16.net004.net",
	       model_path0 = "../models/vgg16.net004.data",
	       net_path1 = "vgg16.net004.net",
	       model_path1 = "vgg16.net004.data";

	Net004 net0;
	parser.read(net_path0, model_path0, &net0);
	parser.write(&net0,net_path1,model_path1);

	Layers & ls0 = net0.ls;
	DataLayer* l0 = (DataLayer*)ls0["data"];
	resize(img,img,Size(l0->outputs[0].h, l0->outputs[0].w));
	l0->add_image((uchar*)img.data,0, 123.68, 116.779, 103.939);
	((DataLayer*)ls0["label"])->add_label(label,0);
	net0.forward();
	float score0 = net0.ls["loss"]->outputs[0].data[0];

	Net004 net1;
	parser.read(net_path1, model_path1, &net1);
	Layers & ls1 = net1.ls;
	DataLayer* l1 = (DataLayer*)ls1["data"];
	l1->add_image((uchar*)img.data,0, 123.68, 116.779, 103.939);
	((DataLayer*)ls1["label"])->add_label(label,0);
	net1.forward();
	float score1 = net1.ls["loss"]->outputs[0].data[0];

	bool ret = abs(score0 - score1) < 1e-5;
	printf("[TEST] parser read/write vgg16 %s\n",ret?"sucessful":"failed");
}
int main(){
	test_cifar10();
	test_vgg10();
	return 0;
}
