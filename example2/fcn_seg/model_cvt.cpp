#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include <map>
using namespace std;
using namespace cv;

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
	if(argc != 2){
		printf("./model_cvt_fcn_seg show\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	map<string,vector<string> > maps;
	maps["fcn8s"] = {
		"../caffe_models/segmentation/fcn8s.prototxt",
		"../caffe_models/segmentation/fcn8s-atonce-pascal.caffemodel",
		"../models2/fcn_seg_8s.net004.net",
		"../models2/fcn_seg_8s.net004.data"
	};
	bool is_show = atoi(argv[1]);
	cvt("fcn8s",maps["fcn8s"],is_show);
	return 0;
}
