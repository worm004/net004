#include "Net004.h"
#include "DataLayer.h"
#include "Parser.h"
#include "caffe_parser.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
using namespace std;
using namespace cv;

void cvt_caffe_model(){
	//string caffe_net_path = "../caffe_models/cifar10_quick_train_test.prototxt",
	//       caffe_model_path = "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5",
	//       net004_net_path = "../models/cifar.net004.net",
	//       net004_model_path = "../models/cifar.net004.data";
	
	//string caffe_net_path = "../caffe_models/VGG_ILSVRC_16_layers_deploy.prototxt",
	//       caffe_model_path = "../caffe_models/VGG_ILSVRC_16_layers.caffemodel",
	//       net004_net_path = "../models/vgg16.net004.net",
	//       net004_model_path = "../models/vgg16.net004.data";

	//string caffe_net_path = "../caffe_models/bvlc_alexnet.prototxt",
	//       caffe_model_path = "../caffe_models/bvlc_alexnet.caffemodel",
	//       net004_net_path = "../models/alexnet.net004.net",
	//       net004_model_path = "../models/alexnet.net004.data";

	string caffe_net_path = "../caffe_models/bvlc_googlenet_deploy.prototxt",
	       caffe_model_path = "../caffe_models/bvlc_googlenet.caffemodel",
	       net004_net_path = "../models/gnetv1.net004.net",
	       net004_model_path = "../models/gnetv1.net004.data";

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
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	cvt_caffe_model();
	return 0;
}
