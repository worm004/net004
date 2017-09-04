#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include <map>
#include "faster_rcnn_tool.h"
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
int main(int argc, char **argv){
	if(argc != 2){
		printf("./model_cvt_faster_rcnn show\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	setting_python_path("/Users/worm004/Projects/py-faster-rcnn/caffe-fast-rcnn/python:/Users/worm004/Projects/py-faster-rcnn/lib");
	map<string,vector<string> > maps;
	maps["faster_rcnn"] = {
		"../caffe_models/detection/faster_rcnn_test.pt",
		"../caffe_models/detection/VGG16_faster_rcnn_final.caffemodel",
		"../models2/faster_rcnn.net004.net",
		"../models2/faster_rcnn.net004.data"
	};
	bool is_show = atoi(argv[1]);
	cvt("faster_rcnn", maps["faster_rcnn"],is_show);
	return 0;
}
