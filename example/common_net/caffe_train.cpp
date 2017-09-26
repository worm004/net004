#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include "glog/logging.h"
using namespace cv;
using namespace std;
int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> ("/Users/worm004/caffe/examples/cifar10/cifar10_quick_train_test.prototxt", caffe::TEST);
	net->CopyTrainedLayersFrom("/Users/worm004/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5");
	net->Forward()[0];

	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();
	const vector<vector<caffe::Blob<float> *> >& bottoms = net->bottom_vecs();
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const caffe::Blob<float>* acc_blob = tops[tops.size()-2][0];

	const caffe::Blob<float>* label_blob = bottoms[tops.size()-2][1];
	const caffe::Blob<float>* predict_blob = bottoms[tops.size()-2][0];
	const caffe::Blob<float>* data_blob = tops[0][0];
	for(int i=0;i<10;++i)
		printf(" %f",data_blob->cpu_data()[i]);
	printf("\n");


	// show predict
	for(int i=0;i<5;++i){
		printf("%f\n",label_blob->cpu_data()[i]);
		for(int j=0;j<10;++j)
			printf(" %f",predict_blob->cpu_data()[i*10+j]);
		printf("\n");
	}

	// run batches
	//for(int i=0;i<10;++i){
	//	net->Forward()[0];
	//	printf("%f\n",acc_blob->cpu_data()[0]);
	//}
	for(int i=0;i<layers.size();++i){
		string layer_type = layers[i]->type();
		printf("%s\n",layer_type.c_str());
	}
	return 0;
}
