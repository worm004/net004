#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
using namespace std;
using namespace cv;

void cvt_caffe_model(const std::string& name){
	string caffe_net_path, caffe_model_path, net004_net_path,net004_model_path;

	if(name == "cifar10"){
	caffe_net_path = "../caffe_models/cifar10_quick_train_test.prototxt",
	caffe_model_path = "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5",
	net004_net_path = "../models/cifar.net004.net",
	net004_model_path = "../models/cifar.net004.data";
	}
	else if(name == "vgg16"){
	caffe_net_path = "../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt",
	caffe_model_path = "../caffe_models/VGG_ILSVRC_16_layers.caffemodel",
	net004_net_path = "../models/vgg16.net004.net",
	net004_model_path = "../models/vgg16.net004.data";
	}
	else if(name == "alexnet"){
	caffe_net_path = "../caffe_models/bvlc_alexnet.prototxt",
	caffe_model_path = "../caffe_models/bvlc_alexnet.caffemodel",
	net004_net_path = "../models/alexnet.net004.net",
	net004_model_path = "../models/alexnet.net004.data";
	}
	else if(name == "gnetv1"){
	caffe_net_path = "../caffe_models/bvlc_googlenet_deploy.prototxt",
	caffe_model_path = "../caffe_models/bvlc_googlenet.caffemodel",
	net004_net_path = "../models/gnetv1.net004.net",
	net004_model_path = "../models/gnetv1.net004.data";
	}
	else if(name == "resnet50"){
	caffe_net_path = "../caffe_models/ResNet-50-deploy.prototxt",
	caffe_model_path = "../caffe_models/ResNet-50-model.caffemodel",
	net004_net_path = "../models/resnet50.net004.net",
	net004_model_path = "../models/resnet50.net004.data";
	}
	else if(name == "resnet101"){
	caffe_net_path = "../caffe_models/ResNet-101-deploy.prototxt",
	caffe_model_path = "../caffe_models/ResNet-101-model.caffemodel",
	net004_net_path = "../models/resnet101.net004.net",
	net004_model_path = "../models/resnet101.net004.data";
	}
	else if(name == "resnet152"){
	caffe_net_path = "../caffe_models/ResNet-152-deploy.prototxt",
	caffe_model_path = "../caffe_models/ResNet-152-model.caffemodel",
	net004_net_path = "../models/resnet152.net004.net",
	net004_model_path = "../models/resnet152.net004.data";
	}
	else if(name == "sqnet1.0"){
	caffe_net_path = "../caffe_models/sqnet1.0.prototxt",
	caffe_model_path = "../caffe_models/sqnet1.0.caffemodel",
	net004_net_path = "../models/sqnet1.0.net004.net",
	net004_model_path = "../models/sqnet1.0.net004.data";
	}
	else if(name == "sqnet1.1"){
	caffe_net_path = "../caffe_models/sqnet1.1.prototxt",
	caffe_model_path = "../caffe_models/sqnet1.1.caffemodel",
	net004_net_path = "../models/sqnet1.1.net004.net",
	net004_model_path = "../models/sqnet1.1.net004.data";
	}
	else{
		printf("no such net: %s\n",name.c_str());
		return;
	}


	CaffeModelParser parser;
	printf("convert caffe model:\n");
	printf("src net: %s\n",caffe_net_path.c_str());
	printf("src model: %s\n",caffe_model_path.c_str());
	printf("des net: %s\n",net004_net_path.c_str());
	printf("des model: %s\n",net004_model_path.c_str());

	parser.load_caffe_model(caffe_net_path, caffe_model_path);
	//parser.show_layers();
	parser.write(net004_net_path, net004_model_path);

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
