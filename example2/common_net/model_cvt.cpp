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
		"../models2/cifar.net004.net",
		"../models2/cifar.net004.data"
		};

	maps["alexnet"] = {
		"../caffe_models/bvlc_alexnet.prototxt",
		"../caffe_models/bvlc_alexnet.caffemodel",
		"../models2/alexnet.net004.net",
		"../models2/alexnet.net004.data"
	};
	
	maps["vgg16"] = {
		"../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt",
		"../caffe_models/VGG_ILSVRC_16_layers.caffemodel",
		"../models2/vgg16.net004.net",
		"../models2/vgg16.net004.data"
	};
	maps["gnetv1"] = {
		"../caffe_models/bvlc_googlenet_deploy.prototxt",
		"../caffe_models/bvlc_googlenet.caffemodel",
		"../models2/gnetv1.net004.net",
		"../models2/gnetv1.net004.data"
	};

	string name = argv[1];
	bool is_show = atoi(argv[2]);
	if(name == "all") for(auto i:maps) cvt(i.first,i.second,is_show);
	else if(maps.find(name) != maps.end()) cvt(name, maps[name],is_show);
	else printf("no such net: %s\n",name.c_str());
	return 0;
}
