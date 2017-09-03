#include <map>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "caffe/caffe.hpp"
#include "Net004.h"
#include "DataLayer.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace cv;
using namespace std;


struct TestParameter{
	TestParameter(){}
	TestParameter(const std::string& cmp, const std::string& cnp, 
			const std::string& nmp, const std::string& nnp,
			const std::string& lp, 
			float mr, float mg, float mb, float sr, float sg, float sb,
			int nl, int ns, int l){
		caffe_model_path = cmp;
		caffe_net_path = cnp;
		net004_model_path = nmp;
		net004_net_path = nnp;
		list_path = lp;
		mean_r = mr;
		mean_g = mg;
		mean_b = mb;
		std_r = sr;
		std_g = sg;
		std_b = sb;
		nlabel = nl;
		nshow = ns;
		label = l;
	}
	
	string caffe_model_path,
		caffe_net_path,
		net004_model_path,
		net004_net_path,
		list_path;

	float mean_r, mean_g, mean_b, std_r, std_g, std_b;
	int nshow, nlabel;
	int label;
};

void show_ret(std::shared_ptr<caffe::Net<float> > net, const std::string& path, int total, int seen){
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const caffe::Blob<float>* blob = tops[tops.size()-2][0];

	ifstream file(path);
	vector<pair<string,float> > labels;
	for(int i=0;i<total;++i){
		string line;
		getline(file,line);
		labels.push_back(make_pair(line,blob->cpu_data()[i]));
	}
	sort(labels.begin(), labels.end(), 
			[](const pair<string,float> & a, const pair<string,float> & b) -> bool
			{ return a.second < b.second; });
	for(int i=total-seen;i<total;++i)
		printf("[%f] %s\n",labels[i].second, labels[i].first.c_str());
}
void show_ret_net004(Net004& net, const std::string& path, int total, int seen){
	ifstream file(path);
	vector<pair<string,float> > labels;
	for(int i=0;i<total;++i){
		string line;
		getline(file,line);
		labels.push_back(make_pair(line,net[net.ls.size()-2]->outputs[0].data[i]));
	}
	sort(labels.begin(), labels.end(), 
			[](const pair<string,float> & a, const pair<string,float> & b) -> bool
			{ return a.second < b.second; });
	for(int i=total-seen;i<total;++i)
		printf("[%f] %s\n",labels[i].second, labels[i].first.c_str());
}
float caffe_forward(const std::string& img_path, const TestParameter& param, bool show){
	if(show) printf("caffe forwarding ...\n");
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (param.caffe_net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(param.caffe_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;
	int c = net->input_blobs()[0]->channels(),
	    h = net->input_blobs()[0]->height(), 
	    w = net->input_blobs()[0]->width();
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data(),
	      * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = param.label;
	Mat img = imread(img_path);
	resize(img,img,Size(h,w));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		input[(i*w+j) + h*w*0] = (data[(i*w+j)*3+2] - param.mean_r)/param.std_r;
		input[(i*w+j) + h*w*1] = (data[(i*w+j)*3+1] - param.mean_g)/param.std_g;
		input[(i*w+j) + h*w*2] = (data[(i*w+j)*3+0] - param.mean_b)/param.std_b;
	}
	net->Forward()[0];
	t1 = now();
	const caffe::Blob<float>* blob = net->Forward()[0];
	t2 = now();
	if(show) {
		cout<<"forward: "<<cal_duration(t1,t2)<<endl;
		show_ret(net,param.list_path,param.nlabel,param.nshow);
	}
	return blob->cpu_data()[0];
}
float net004_forward(const std::string& img_path, const TestParameter& param, bool show){
	Net004 net;
	auto t1 = now();
	net.load(param.net004_net_path, param.net004_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	DataLayer* l0 = (DataLayer*)net["input_data"], *l1 = (DataLayer*)net["input_label"];
	int h = l0->h, w = l0->w;
	l0->n = l1->n = 1;
	l1->c = l1->h = l1->w = 1;

	net.pre_alloc();
	l1->outputs[0].data[0] = param.label;
	Mat img = imread(img_path);
	resize(img,img,Size(h, w));
	uchar *idata = (uchar*)img.data;
	float * data = l0->outputs[0].data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		data[(i*w+j) + h*w*0] = (idata[(i*w+j)*3+2] - param.mean_r)/param.std_r;
		data[(i*w+j) + h*w*1] = (idata[(i*w+j)*3+1] - param.mean_g)/param.std_g;
		data[(i*w+j) + h*w*2] = (idata[(i*w+j)*3+0] - param.mean_b)/param.std_b;
	}
	//net.show();
	t1 = now();
	net.forward();
	t2 = now();
	if(show) {
		cout<<"forward: "<<cal_duration(t1,t2)<<endl;
		show_ret_net004(net,param.list_path,param.nlabel,param.nshow);
	}
	return net[net.ls.size()-1]->outputs[0].data[0];
}


void test(const std::string& img_path, const std::string& net_name, const TestParameter & param, bool show){
	printf("[TEST] [forwrad] %s\n", net_name.c_str());
	float caffe_score = caffe_forward(img_path, param, show);
	float net004_score = net004_forward(img_path, param, show);
	bool ret = abs(caffe_score - net004_score) < 1e-5;
	if(show) printf("caffe score: %f\nnet004 score: %f\n",caffe_score,net004_score);
	printf("[TEST] [result] %s\n",ret?"sucessful":"failed");
}
int main(int argc, char **argv){
	if(argc !=3){
		printf("./net_test model_name/all 0/1\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	map<string, TestParameter> maps;
	maps["cifar10"] = TestParameter(
	       "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5",
	       "../caffe_models/cifar10_quick_train_test.prototxt",
	       "../models2/cifar.net004.data",
	       "../models2/cifar.net004.net",
	       "../caffe_models/cifar10.list",
	       127,127,127,
	       1,1,1,
	       10,5, 8
	);
	maps["alexnet"] = TestParameter(
	       "../caffe_models/bvlc_alexnet.caffemodel",
	       "../caffe_models/bvlc_alexnet.prototxt",
	       "../models2/alexnet.net004.data",
	       "../models2/alexnet.net004.net",
	       "../caffe_models/imagenet2012.list",
	       123.68,116.779,103.939,
	       1,1,1,
	       1000,5, 628
	);
	maps["vgg16"] = TestParameter(
	       "../caffe_models/VGG_ILSVRC_16_layers.caffemodel",
	       "../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt",
	       "../models2/vgg16.net004.data",
	       "../models2/vgg16.net004.net",
	       "../caffe_models/imagenet2012.list",
	       123.68,116.779,103.939,
	       1,1,1,
	       1000,5, 628
	);
	maps["gnetv1"] = TestParameter(
	       "../caffe_models/bvlc_googlenet.caffemodel",
	       "../caffe_models/bvlc_googlenet_deploy.prototxt",
	       "../models2/gnetv1.net004.data",
	       "../models2/gnetv1.net004.net",
	       "../caffe_models/imagenet2012.list",
	       123.68,116.779,103.939,
	       1,1,1,
	       1000,5, 628
	);
	//maps["resnet50"] = TestParameter(
	//       "../caffe_models/ResNet-50-model.caffemodel",
	//       "../caffe_models/ResNet-50-deploy.prototxt",
	//       "../models/resnet50.net004.data",
	//       "../models/resnet50.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["resnet101"] = TestParameter(
	//       "../caffe_models/ResNet-101-model.caffemodel",
	//       "../caffe_models/ResNet-101-deploy.prototxt",
	//       "../models/resnet101.net004.data",
	//       "../models/resnet101.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["resnet152"] = TestParameter(
	//       "../caffe_models/ResNet-152-model.caffemodel",
	//       "../caffe_models/ResNet-152-deploy.prototxt",
	//       "../models/resnet152.net004.data",
	//       "../models/resnet152.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["sqnet1.0"] = TestParameter(
	//       "../caffe_models/sqnet1.0.caffemodel",
	//       "../caffe_models/sqnet1.0.prototxt",
	//       "../models/sqnet1.0.net004.data",
	//       "../models/sqnet1.0.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["sqnet1.1"] = TestParameter(
	//       "../caffe_models/sqnet1.1.caffemodel",
	//       "../caffe_models/sqnet1.1.prototxt",
	//       "../models/sqnet1.1.net004.data",
	//       "../models/sqnet1.1.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["sqnet_res"] = TestParameter(
	//       "../caffe_models/sqnet_res.caffemodel",
	//       "../caffe_models/sqnet_res.prototxt",
	//       "../models/sqnet_res.net004.data",
	//       "../models/sqnet_res.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       123.68,116.779,103.939,
	//       1,1,1,
	//       1000,5, 628
	//);
	//maps["gnetv3"] = TestParameter(
	//       "../caffe_models/inception-v3.caffemodel",
	//       "../caffe_models/deploy_inception-v3.prototxt",
	//       "../models/gnetv3.net004.data",
	//       "../models/gnetv3.net004.net",
	//       "../caffe_models/imagenet2015.list",
	//       128,128,128,
	//       128,128,128,
	//       1000,5, 243
	//);
	//maps["gnetv4"] = TestParameter(
	//       "../caffe_models/inception-v4.caffemodel",
	//       "../caffe_models/deploy_inception-v4.prototxt",
	//       "../models/gnetv4.net004.data",
	//       "../models/gnetv4.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       128,128,128,
	//       128,128,128,
	//       1000,5, 628
	//);
	//maps["dense121"] = TestParameter(
	//       "../caffe_models/DenseNet_121.caffemodel",
	//       "../caffe_models/DenseNet_121.prototxt",
	//       "../models/dense121.net004.data",
	//       "../models/dense121.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       128,128,128,
	//       128,128,128,
	//       1000,5, 628
	//);
	//maps["in-res-v2"] = TestParameter(
	//       "../caffe_models/inception-resnet-v2.caffemodel",
	//       "../caffe_models/deploy_inception-resnet-v2.prototxt",
	//       "../models/inception-res-v2.net004.data",
	//       "../models/inception-res-v2.net004.net",
	//       "../caffe_models/imagenet2012.list",
	//       128,128,128,
	//       128,128,128,
	//       1000,5, 628
	//);

	string name = argv[1];
	bool show = atoi(argv[2]);
	string img_path = "../imgs/westerdam-ship-size.jpg";
	//string img_path = "../imgs/tabby-cat-names.jpg";
	if(name == string("all")) for(auto i: maps) test(img_path, i.first, i.second,show);
	else if(maps.find(name) != maps.end()) test(img_path, name, maps[name],show);
	else printf("no such net: %s\n",argv[1]);
	return 0;
}
