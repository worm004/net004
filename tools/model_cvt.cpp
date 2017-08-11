#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include <map>
using namespace std;
using namespace cv;

void cvt_caffe_model(const std::string& name){
	string caffe_net_path, caffe_model_path, net004_net_path, net004_model_path;
	map<string,vector<string> > maps;

	caffe_net_path = "../caffe_models/cifar10_quick_train_test.prototxt";
	caffe_model_path = "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5";
	net004_net_path = "../models/cifar.net004.net";
	net004_model_path = "../models/cifar.net004.data";
	maps["cifar10"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt";
	caffe_model_path = "../caffe_models/VGG_ILSVRC_16_layers.caffemodel";
	net004_net_path = "../models/vgg16.net004.net";
	net004_model_path = "../models/vgg16.net004.data";
	maps["vgg16"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/bvlc_alexnet.prototxt";
	caffe_model_path = "../caffe_models/bvlc_alexnet.caffemodel";
	net004_net_path = "../models/alexnet.net004.net";
	net004_model_path = "../models/alexnet.net004.data";
	maps["alexnet"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/bvlc_googlenet_deploy.prototxt";
	caffe_model_path = "../caffe_models/bvlc_googlenet.caffemodel";
	net004_net_path = "../models/gnetv1.net004.net";
	net004_model_path = "../models/gnetv1.net004.data";
	maps["gnetv1"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/ResNet-50-deploy.prototxt";
	caffe_model_path = "../caffe_models/ResNet-50-model.caffemodel";
	net004_net_path = "../models/resnet50.net004.net";
	net004_model_path = "../models/resnet50.net004.data";
	maps["resnet50"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/ResNet-101-deploy.prototxt";
	caffe_model_path = "../caffe_models/ResNet-101-model.caffemodel";
	net004_net_path = "../models/resnet101.net004.net";
	net004_model_path = "../models/resnet101.net004.data";
	maps["resnet101"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/ResNet-152-deploy.prototxt";
	caffe_model_path = "../caffe_models/ResNet-152-model.caffemodel";
	net004_net_path = "../models/resnet152.net004.net";
	net004_model_path = "../models/resnet152.net004.data";
	maps["resnet152"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/sqnet1.0.prototxt";
	caffe_model_path = "../caffe_models/sqnet1.0.caffemodel";
	net004_net_path = "../models/sqnet1.0.net004.net";
	net004_model_path = "../models/sqnet1.0.net004.data";
	maps["sqnet1.0"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/sqnet1.1.prototxt";
	caffe_model_path = "../caffe_models/sqnet1.1.caffemodel";
	net004_net_path = "../models/sqnet1.1.net004.net";
	net004_model_path = "../models/sqnet1.1.net004.data";
	maps["sqnet1.1"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/sqnet_res.prototxt";
	caffe_model_path = "../caffe_models/sqnet_res.caffemodel";
	net004_net_path = "../models/sqnet_res.net004.net";
	net004_model_path = "../models/sqnet_res.net004.data";
	maps["sqnet_res"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/deploy_inception-v3.prototxt";
	caffe_model_path = "../caffe_models/inception-v3.caffemodel";
	net004_net_path = "../models/gnetv3.net004.net";
	net004_model_path = "../models/gnetv3.net004.data";
	maps["gnetv3"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	caffe_net_path = "../caffe_models/deploy_inception-v4.prototxt";
	caffe_model_path = "../caffe_models/inception-v4.caffemodel";
	net004_net_path = "../models/gnetv4.net004.net";
	net004_model_path = "../models/gnetv4.net004.data";
	maps["gnetv4"] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	if(name == "all"){
		for(auto i:maps){
			printf("convert caffe model:\n");
			printf("src net: %s\n",i.second[0].c_str());
			printf("src model: %s\n",i.second[1].c_str());
			printf("des net: %s\n",i.second[2].c_str());
			printf("des model: %s\n",i.second[3].c_str());
			CaffeModelParser parser;
			parser.load_caffe_model(i.second[0], i.second[1]);
			parser.write(i.second[2],i.second[3]);
		}
	}
	else if(maps.find(name) != maps.end()){
		printf("convert caffe model:\n");
		printf("src net: %s\n",maps[name][0].c_str());
		printf("src model: %s\n",maps[name][1].c_str());
		printf("des net: %s\n",maps[name][2].c_str());
		printf("des model: %s\n",maps[name][3].c_str());
		CaffeModelParser parser;
		parser.load_caffe_model(maps[name][0], maps[name][1]);
		parser.write(maps[name][2], maps[name][3]);
	}
	else{
		printf("no such net: %s\n",name.c_str());
		return;
	}

}
int main(int argc, char **argv){
	if(argc != 2){
		printf("1 param is needed\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	cvt_caffe_model(argv[1]);
	return 0;
}
