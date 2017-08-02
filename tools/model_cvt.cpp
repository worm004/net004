#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

void cvt_caffe_model(){
	string net_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_train_test.prototxt";
	string model_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_iter_5000.caffemodel.h5";
	CaffeModelParser parser;
	parser.load_caffe_model(net_path, model_path);
	parser.show_layers();
	parser.write("../models/cifar_test.net004.net","../models/cifar_test.net004.data");

}
void test_caffe_model(){
	Net004 net;
	Parser parser0;
	parser0.read("../models/cifar_test.net004.net", "../models/cifar_test.net004.data", &net);

	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread("/Users/worm004/Projects/net004/caffe_example/westerdam-ship-size.jpg");
	resize(img,img,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)img.data,0);

	DataLayer* l2 = (DataLayer*)ls["label"];
	l2->add_label(8,0);

	net.forward();
	Layer * ll = net.ls["loss"];
	for(int i=0;i<ll->outputs[0].n;++i){
		for(int j=0;j<ll->outputs[0].chw();++j)
			printf("%f ",ll->outputs[0].data[j + i * ll->outputs[0].chw()]);
		printf("\n");
	}
}
int main(){
	//cvt_caffe_model();
	test_caffe_model();
	return 0;
}
