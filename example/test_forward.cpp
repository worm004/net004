#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include <string>
#include <string>
#include <fstream>
#include <iostream>
#include "Net004.h"
#include "ConvLayer.h"
#include "DataLayer.h"
#include "FCLayer.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;
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
void load_img(Net004& net){
	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread("/Users/worm004/Projects/net004/caffe_example/westerdam-ship-size.jpg");
	resize(img,img,Size(l->outputs[0].h, l->outputs[0].w));
	l->load_image((uchar*)img.data);
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
	cout<<"conv0: "<<model[0][0].size()<<" : "<<conv0->weight.total()<<" "<<model[0][1].size()<<" : "<<conv0->bias.total()<<endl;
	for(int i=0;i<model[0][0].size();++i)
		weight_data[i] = model[0][0][i];
	for(int i=0;i<model[0][1].size();++i)
		bias_data[i] = model[0][1][i];

	ConvLayer * conv1 = (ConvLayer*)ls["conv1"];
	float * weight1_data = conv1->weight.data;
	float * bias1_data = conv1->bias.data;
	cout<<"conv1: "<<model[1][0].size()<<" : "<<conv1->weight.total()<<" "<<model[1][1].size()<<" : "<<conv1->bias.total()<<endl;
	for(int i=0;i<model[1][0].size();++i)
		weight1_data[i] = model[1][0][i];
	for(int i=0;i<model[1][1].size();++i)
		bias1_data[i] = model[1][1][i];

	ConvLayer * conv2 = (ConvLayer*)ls["conv2"];
	float * weight2_data = conv2->weight.data;
	float * bias2_data = conv2->bias.data;
	cout<<"conv2: "<<model[2][0].size()<<" : "<<conv2->weight.total()<<" "<<model[2][1].size()<<" : "<<conv2->bias.total()<<endl;
	for(int i=0;i<model[2][0].size();++i)
		weight2_data[i] = model[2][0][i];
	for(int i=0;i<model[2][1].size();++i)
		bias2_data[i] = model[2][1][i];
		
	FCLayer * fc0 = (FCLayer*)ls["fc0"];
	float * weight3_data = fc0->weight.data;
	float * bias3_data = fc0->bias.data;
	cout<<"fc0: "<<model[3][0].size()<<" : "<<fc0->weight.total()<<" "<<model[3][1].size()<<" : "<<fc0->bias.total()<<endl;
	for(int i=0;i<model[3][0].size();++i)
		weight3_data[i] = model[3][0][i];
	for(int i=0;i<model[3][1].size();++i)
		bias3_data[i] = model[3][1][i];

	FCLayer * fc1 = (FCLayer*)ls["fc1"];
	float * weight4_data = fc1->weight.data;
	float * bias4_data = fc1->bias.data;
	cout<<"fc1: "<<model[4][0].size()<<" : "<<fc1->weight.total()<<" "<<model[4][1].size()<<" : "<<fc1->bias.total()<<endl;
	for(int i=0;i<model[4][0].size();++i)
		weight4_data[i] = model[4][0][i];
	for(int i=0;i<model[4][1].size();++i)
		bias4_data[i] = model[4][1][i];

}
void test_cifar10(){
	Net004 net("cifar10");
	Layers & ls = net.ls;
	ls.add_data("data",1,3,32,32,"image");
	ls.add_conv("conv0",{32,5,1,2},"");
	ls.add_activity("relu0","relu");
	ls.add_pool("maxpool0",{3,2,0},"max");
	ls.add_conv("conv1",{32,5,1,2},"relu");
	ls.add_pool("avgpool0",{3,2,0},"avg");
	ls.add_conv("conv2",{64,5,1,2},"relu");
	ls.add_pool("avgpool1",{3,2,0},"avg");
	ls.add_fc("fc0",64,"");
	ls.add_fc("fc1",10,"");
	//ls.show();
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
			"fc1"});
	cs.add(t0);
	cs.update();
	//cs.show();
	net.check();
	net.setup();
	load_model(net);
	load_img(net);
	//net.show();
	
	net.forward();
	Layer * l = ls["fc1"];
	for(int i=0;i<l->outputs[0].total();++i)
		printf("%f ",l->outputs[0].data[i]);
	printf("\n");
}

int main(){
	test_cifar10();
	return 0;
}
