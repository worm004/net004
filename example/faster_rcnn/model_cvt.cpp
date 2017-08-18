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

void cvt_caffe_model(const std::string& name){
	string caffe_net_path, caffe_model_path, net004_net_path, net004_model_path;
	map<string,vector<string> > maps;
	
	caffe_net_path = "../caffe_models/detection/faster_rcnn_test.pt";
	caffe_model_path = "../caffe_models/detection/VGG16_faster_rcnn_final.caffemodel";
	net004_net_path = "../models/detection/faster_rcnn.net004.net";
	net004_model_path = "../models/detection/faster_rcnn.net004.data";
	maps[name] = {caffe_net_path, caffe_model_path,net004_net_path,net004_model_path};

	printf("convert caffe model:\n");
	printf("src net: %s\n",maps[name][0].c_str());
	printf("src model: %s\n",maps[name][1].c_str());
	printf("des net: %s\n",maps[name][2].c_str());
	printf("des model: %s\n",maps[name][3].c_str());
	CaffeModelParser parser;
	parser.load_caffe_model(maps[name][0], maps[name][1]);
	parser.write(maps[name][2], maps[name][3]);

}
int main(int argc, char **argv){

	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	setting_python_path("/Users/worm004/Projects/py-faster-rcnn/caffe-fast-rcnn/python:/Users/worm004/Projects/py-faster-rcnn/lib");

	cvt_caffe_model("faster_rcnn");
	return 0;
}
