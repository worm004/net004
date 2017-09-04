#include <string>
#include <map>
#include <vector>
#include "caffe_parser.h"
#include "glog/logging.h"
using namespace std;

void cvt(const std::string&net_name, const std::vector<std::string>& ps,bool show){
	printf("[convert] %s: caffe --> net004 ",net_name.c_str());
	if(show){
		printf("\nsrc net: %s\n",ps[0].c_str());
		printf("src model: %s\n",ps[1].c_str());
		printf("des net: %s\n",ps[2].c_str());
		printf("des model: %s\n",ps[3].c_str());
	}
	CaffeParser parser;
	bool is_train = false;
	parser.load_caffe_model(ps[0], ps[1], is_train);
	parser.convert();
	parser.write(ps[2], ps[3]);
	printf("[sucessful]\n");
}
int main(int argc, char**argv){
	if(argc != 3){
		printf("./model_cvt name show\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	map<string,vector<string> > maps;
	maps["cifar10"] = {
		"../caffe_models/cifar10_quick_train_test.prototxt",
		"../caffe_models/cifar10_quick_iter_5000.caffemodel.h5",
		"../models/cifar.net004.net",
		"../models/cifar.net004.data"
		};
	maps["alexnet"] = {
		"../caffe_models/bvlc_alexnet.prototxt",
		"../caffe_models/bvlc_alexnet.caffemodel",
		"../models/alexnet.net004.net",
		"../models/alexnet.net004.data"
	};
	maps["vgg16"] = {
		"../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt",
		"../caffe_models/VGG_ILSVRC_16_layers.caffemodel",
		"../models/vgg16.net004.net",
		"../models/vgg16.net004.data"
	};
	maps["gnetv1"] = {
		"../caffe_models/bvlc_googlenet_deploy.prototxt",
		"../caffe_models/bvlc_googlenet.caffemodel",
		"../models/gnetv1.net004.net",
		"../models/gnetv1.net004.data"
	};
	maps["resnet50"] = {
		"../caffe_models/ResNet-50-deploy.prototxt",
		"../caffe_models/ResNet-50-model.caffemodel",
		"../models/resnet50.net004.net",
		"../models/resnet50.net004.data"
	};
	maps["resnet101"] = {
		"../caffe_models/ResNet-101-deploy.prototxt",
		"../caffe_models/ResNet-101-model.caffemodel",
		"../models/resnet101.net004.net",
		"../models/resnet101.net004.data"
	};
	maps["resnet152"] = {
		"../caffe_models/ResNet-152-deploy.prototxt",
		"../caffe_models/ResNet-152-model.caffemodel",
		"../models/resnet152.net004.net",
		"../models/resnet152.net004.data"
	};
	maps["sqnet1.0"] = {
		"../caffe_models/sqnet1.0.prototxt",
		"../caffe_models/sqnet1.0.caffemodel",
		"../models/sqnet1.0.net004.net",
		"../models/sqnet1.0.net004.data"
	};
	maps["sqnet1.1"] = {
		"../caffe_models/sqnet1.1.prototxt",
		"../caffe_models/sqnet1.1.caffemodel",
		"../models/sqnet1.1.net004.net",
		"../models/sqnet1.1.net004.data"
	};
	maps["sqnet_res"] = {
		"../caffe_models/sqnet_res.prototxt",
		"../caffe_models/sqnet_res.caffemodel",
		"../models/sqnet_res.net004.net",
		"../models/sqnet_res.net004.data"
	};
	maps["gnetv3"] = {
		"../caffe_models/deploy_inception-v3.prototxt",
		"../caffe_models/inception-v3.caffemodel",
		"../models/gnetv3.net004.net",
		"../models/gnetv3.net004.data"
	};
	maps["gnetv4"] = {
		"../caffe_models/deploy_inception-v4.prototxt",
		"../caffe_models/inception-v4.caffemodel",
		"../models/gnetv4.net004.net",
		"../models/gnetv4.net004.data"
	};
	maps["dense121"] = {
		"../caffe_models/DenseNet_121.prototxt",
		"../caffe_models/DenseNet_121.caffemodel",
		"../models/dense121.net004.net",
		"../models/dense121.net004.data"
	};
	maps["in-res-v2"] = {
		"../caffe_models/deploy_inception-resnet-v2.prototxt",
		"../caffe_models/inception-resnet-v2.caffemodel",
		"../models/inception-res-v2.net004.net",
		"../models/inception-res-v2.net004.data"
	};

	string name = argv[1];
	bool is_show = atoi(argv[2]);
	if(name == "all") for(auto i:maps) cvt(i.first,i.second,is_show);
	else if(maps.find(name) != maps.end()) cvt(name, maps[name],is_show);
	else printf("no such net: %s\n",name.c_str());
	return 0;
}
