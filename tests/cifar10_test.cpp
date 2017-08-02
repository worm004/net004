#include <string>
#include "caffe/caffe.hpp"
#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
using namespace cv;
using namespace std;
float caffe_forward(const std::string& img_path, int label){
	string net_path = "../caffe_models/cifar10_quick_train_test.prototxt",
	       model_path = "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5";
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	net->input_blobs()[0]->Reshape(1, 3, 32, 32);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data(), 
	      * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = label;
	Mat img = imread(img_path);
	resize(img,img,Size(32,32));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<32;++i)
	for(int j=0;j<32;++j){
		input[(i*32+j) + 32*32*0] = data[(i*32+j)*3+2] - 127;
		input[(i*32+j) + 32*32*1] = data[(i*32+j)*3+1] - 127;
		input[(i*32+j) + 32*32*2] = data[(i*32+j)*3+0] - 127;
	}
	const caffe::Blob<float>* blob = net->Forward()[0];
	return blob->cpu_data()[0];
}
float net004_forward(const std::string& img_path, int label){
	string net_path = "../models/cifar.net004.net",
	       model_path = "../models/cifar.net004.data";
	Net004 net;
	Parser parser;
	parser.read(net_path, model_path, &net);
	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread(img_path);
	resize(img,img,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)img.data,0);
	((DataLayer*)ls["label"])->add_label(label,0);
	net.forward();
	return net.ls["loss"]->outputs[0].data[0];
}
int main(int argc, char **argv){
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	string img_path = "../imgs/westerdam-ship-size.jpg";
	int label = 8;
	float caffe_score = caffe_forward(img_path, label);
	float net004_score = net004_forward(img_path, label);

	bool ret = abs(caffe_score - net004_score) < 1e-5;
	printf("[TEST] cifar10 test %s\n",ret?"sucessful":"failed");

	return 0;
}
