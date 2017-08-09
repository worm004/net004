#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "caffe/caffe.hpp"
#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace cv;
using namespace std;
float caffe_forward(const std::string& img_path, int label, const std::string& net_name){
	string net_path, model_path;

	if(net_name == "1.0"){
		net_path = "../caffe_models/sqnet1.0.prototxt";
		model_path = "../caffe_models/sqnet1.0.caffemodel";
	} else if(net_name == "1.1"){
		net_path = "../caffe_models/sqnet1.1.prototxt";
		model_path = "../caffe_models/sqnet1.1.caffemodel";
	}

  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	auto t2 = now();
	cout<<"read: "<<cal_duration(t1,t2)<<endl;

	int c = net->input_blobs()[0]->channels(),
	    h = net->input_blobs()[0]->height(), 
	    w = net->input_blobs()[0]->width();
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data(),
	      * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = label;
	Mat img = imread(img_path);
	resize(img,img,Size(h,w));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		input[(i*w+j) + h*w*0] = data[(i*w+j)*3+2] - 123.68;
		input[(i*w+j) + h*w*1] = data[(i*w+j)*3+1] - 116.779;
		input[(i*w+j) + h*w*2] = data[(i*w+j)*3+0] - 103.939;
	}
	t1 = now();
	const caffe::Blob<float>* blob = net->Forward()[0];
	t2 = now();
	cout<<"forward: "<<cal_duration(t1,t2)<<endl;

	//ifstream file("../caffe_models/imagenet.list");
	//vector<pair<string,float> > labels;
	//for(int i=0;i<1000;++i){
	//	string line;
	//	getline(file,line);
	//	labels.push_back(make_pair(line,blob->cpu_data()[i]));
	//}
	//sort(labels.begin(), labels.end(), 
	//		[](const pair<string,float> & a, const pair<string,float> & b) -> bool
	//		{ return a.second < b.second; });
	//for(int i=990;i<1000;++i)
	//	printf("[%f] %s\n",labels[i].second, labels[i].first.c_str());

	return blob->cpu_data()[0];
}
float net004_forward(const std::string& img_path, int label, const std::string& net_name){
	string net_path, model_path;
	
	if(net_name == "1.0"){
		net_path = "../models/sqnet1.0.net004.net";
		model_path = "../models/sqnet1.0.net004.data";
	}else if(net_name == "1.1"){
		net_path = "../models/sqnet1.1.net004.net";
		model_path = "../models/sqnet1.1.net004.data";
	}

	Net004 net;
	Parser parser;
	auto t1 = now();
	parser.read(net_path, model_path, &net);
	auto t2 = now();
	cout<<"read: "<<cal_duration(t1,t2)<<endl;
	//net.show();
	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread(img_path);
	resize(img,img,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)img.data,0, 123.68, 116.779, 103.939);
	((DataLayer*)ls["label"])->add_label(label,0);
	//net.show();

	t1 = now();
	net.forward();
	t2 = now();

	cout<<"forward: "<<cal_duration(t1,t2)<<endl;

	//ifstream file("../caffe_models/imagenet.list");
	//vector<pair<string,float> > labels;
	//for(int i=0;i<1000;++i){
	//	string line;
	//	getline(file,line);
	//	labels.push_back(make_pair(line,net.ls["fc8"]->outputs[0].data[i]));
	//}
	//sort(labels.begin(), labels.end(), 
	//		[](const pair<string,float> & a, const pair<string,float> & b) -> bool
	//		{ return a.second < b.second; });
	//for(int i=990;i<1000;++i)
	//	printf("[%f] %s\n",labels[i].second, labels[i].first.c_str());

	return net.ls["loss"]->outputs[0].data[0];
}
int main(int argc, char **argv){
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	string img_path = "../imgs/westerdam-ship-size.jpg";
	int label = 628;
	char nets[2][100] = {"1.0","1.1"};

	for(int i=0;i<2;++i){
		float caffe_score = caffe_forward(img_path, label,nets[i]);
		float net004_score = net004_forward(img_path, label,nets[i]);
		bool ret = abs(caffe_score - net004_score) < 1e-5;
		printf("[TEST] resnet%s test %s\n",nets[i],ret?"sucessful":"failed");
	}

	return 0;
}
