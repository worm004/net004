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
int main(int argc, char **argv){
	if(argc != 2){
		printf("./model_cvt_yolo show\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	map<string,vector<string> > maps;
	maps["yolov1"] = {
		"../caffe_models/detection/gnet_deploy.prototxt",
		"../caffe_models/detection/gnet_yolo_iter_32000.caffemodel",
		"../models/yolov1.net004.net",
		"../models/yolov1.net004.data"
	};
	bool is_show = atoi(argv[1]);
	cvt("yolo1", maps["yolov1"],is_show);
	return 0;
}
