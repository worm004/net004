#include "opencv2/opencv.hpp"
#include "Parser.h"
#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include <string>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include "Net004.h"
#include "ConvLayer.h"
#include "DataLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"

#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())

using namespace std;
using namespace cv;
void load_img(Net004& net){
	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread("/Users/worm004/Projects/net004/caffe_example/westerdam-ship-size.jpg");
	resize(img,img,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)img.data,0);

	DataLayer* l2 = (DataLayer*)ls["label"];
	l2->add_label(8,0);
	//l->add_image((uchar*)img.data,1);
}
void load_txt_model(const string& path, vector< vector<vector<float> > > & model, vector<string> & names){
	ifstream file(path);
	while(1){
		string name;
		int n = 0;
		file>>name>>n;
		if(file.eof())break;
		if(n == 0) continue;
		names.push_back(name);
		model.resize(model.size()+1);
		vector<vector<float> > & m = model.back();
		m.resize(n);

		//cout<<name<<" "<<n<<endl;
		for(int i=0;i<n;++i){
			string line;
			getline(file,line);
			getline(file,line);
			//cout<<line<<endl;
			int num = atoi(line.substr(line.find_first_of("(")+1, line.find_first_of(")") - line.find_first_of("(")-1).c_str());
			//cout<<"num: "<<num<<endl;
			for(int j=0;j<num;++j){
				file >> line;
				m[i].push_back(atof(line.c_str()));
			}
			//cout<<m[i].size()<<endl;
		}
		//getchar();
	}
}
void load_model(Net004& net){
	string path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_iter_5000.caffemodel.txt";
	vector< vector<vector<float> > > model;
	vector<string> names;
	load_txt_model(path, model, names);

	//for(int i=0;i<names.size();++i){
	//	cout<<names[i]<<" "<<model[i].size()<<endl;
	//	for(int j = 0;j<model[i].size();++j)
	//		cout<<model[i][j].size()<<endl;
	//}
	
	printf("loading ... \n");
	Layers & ls = net.ls;
	ConvLayer * conv0 = (ConvLayer*)ls["conv0"];
	float * weight_data = conv0->weight.data;
	float * bias_data = conv0->bias.data;
	cout<<"conv0: "<<model[0][0].size()<<" : "<<conv0->weight.nchw()<<" "<<model[0][1].size()<<" : "<<conv0->bias.nchw()<<endl;
	for(int i=0;i<model[0][0].size();++i)
		weight_data[i] = model[0][0][i];
	for(int i=0;i<model[0][1].size();++i)
		bias_data[i] = model[0][1][i];

	ConvLayer * conv1 = (ConvLayer*)ls["conv1"];
	float * weight1_data = conv1->weight.data;
	float * bias1_data = conv1->bias.data;
	cout<<"conv1: "<<model[1][0].size()<<" : "<<conv1->weight.nchw()<<" "<<model[1][1].size()<<" : "<<conv1->bias.nchw()<<endl;
	for(int i=0;i<model[1][0].size();++i)
		weight1_data[i] = model[1][0][i];
	for(int i=0;i<model[1][1].size();++i)
		bias1_data[i] = model[1][1][i];

	ConvLayer * conv2 = (ConvLayer*)ls["conv2"];
	float * weight2_data = conv2->weight.data;
	float * bias2_data = conv2->bias.data;
	cout<<"conv2: "<<model[2][0].size()<<" : "<<conv2->weight.nchw()<<" "<<model[2][1].size()<<" : "<<conv2->bias.nchw()<<endl;
	for(int i=0;i<model[2][0].size();++i)
		weight2_data[i] = model[2][0][i];
	for(int i=0;i<model[2][1].size();++i)
		bias2_data[i] = model[2][1][i];
		
	FCLayer * fc0 = (FCLayer*)ls["fc0"];
	float * weight3_data = fc0->weight.data;
	float * bias3_data = fc0->bias.data;
	cout<<"fc0: "<<model[3][0].size()<<" : "<<fc0->weight.nchw()<<" "<<model[3][1].size()<<" : "<<fc0->bias.nchw()<<endl;
	for(int i=0;i<model[3][0].size();++i)
		weight3_data[i] = model[3][0][i];
	for(int i=0;i<model[3][1].size();++i)
		bias3_data[i] = model[3][1][i];

	FCLayer * fc1 = (FCLayer*)ls["fc1"];
	float * weight4_data = fc1->weight.data;
	float * bias4_data = fc1->bias.data;
	cout<<"fc1: "<<model[4][0].size()<<" : "<<fc1->weight.nchw()<<" "<<model[4][1].size()<<" : "<<fc1->bias.nchw()<<endl;
	for(int i=0;i<model[4][0].size();++i)
		weight4_data[i] = model[4][0][i];
	for(int i=0;i<model[4][1].size();++i)
		bias4_data[i] = model[4][1][i];
}
void test_txt2model_cifar10(){
	Net004 net("cifar10");
	Layers & ls = net.ls;
	int batch_size = 1;
	ls.add_data("data",batch_size,3,32,32,"image");
	ls.add_data("label",batch_size,1,1,1,"label");
	ls.add_conv("conv0",{32,5,1,2},"");
	ls.add_activity("relu0","relu");
	ls.add_pool("maxpool0",{3,2,0},"max");
	ls.add_conv("conv1",{32,5,1,2},"relu");
	ls.add_pool("avgpool0",{3,2,0},"avg");
	ls.add_conv("conv2",{64,5,1,2},"relu");
	ls.add_pool("avgpool1",{3,2,0},"avg");
	ls.add_fc("fc0",64,"");
	ls.add_fc("fc1",10,"");
	ls.add_loss("loss","softmax");
	Connections& cs = net.cs;
	vector<string> t0({
			"data",
			"conv0",
			"maxpool0",
			"relu0",
			"conv1",
			"avgpool0",
			"conv2",
			"avgpool1",
			"fc0",
			"fc1",
			"loss"});
	vector<string> tlabel({"label","loss"});
	cs.add(tlabel).add(t0);
	cs.update();
	net.check();
	net.setup();
	load_model(net);

	//ls.show();
	//cs.show();
	//net.show();
	//Parser parser;
	//parser.write(&net, "../models/cifar.net004.net", "../models/cifar.net004.data");
	
	load_img(net);
	net.forward();
	Layer * l = ls["loss"];

	//for(int i=0;i<l->inputs[0].n;++i){
	//	for(int j=0;j<l->inputs[0].chw();++j)
	//		printf("%f ",l->inputs[0].data[j + i * l->inputs[0].chw()]);
	//	printf("\n");
	//}
	for(int i=0;i<l->outputs[0].n;++i){
		for(int j=0;j<l->outputs[0].chw();++j)
			printf("%f ",l->outputs[0].data[j + i * l->outputs[0].chw()]);
		printf("\n");
	}
}
void test_model_cifar10(){
	Parser parser;
	Net004 net("cifar10");
	parser.read("../models/cifar.net004.net", "../models/cifar.net004.data", &net);

	Layers & ls = net.ls;
	Connections& cs = net.cs;
	//ls.show();
	//cs.show();
	//net.show();

	load_img(net);
	net.forward();

	Layer * l = ls["loss"];
	//for(int i=0;i<l->inputs[0].n;++i){
	//	for(int j=0;j<l->inputs[0].chw();++j)
	//		printf("%f ",l->inputs[0].data[j + i * l->inputs[0].chw()]);
	//	printf("\n");
	//}
	for(int i=0;i<l->outputs[0].n;++i){
		for(int j=0;j<l->outputs[0].chw();++j)
			printf("%f ",l->outputs[0].data[j + i * l->outputs[0].chw()]);
		printf("\n");
	}
}
int main(){
	//test_txt2model_cifar10();
	test_model_cifar10();
	return 0;
}
