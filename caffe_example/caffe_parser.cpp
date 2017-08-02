#include "stdio.h"
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"

#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())

using namespace std;
using namespace cv;

class CaffeModelParser{
public:
	void load_caffe_model(const std::string& net_path, const std::string& model_path){
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
		net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
		net->CopyTrainedLayersFrom(model_path);
	}
	void write(const std::string& net_path, const std::string& model_path){
		write_net(net_path);
	}
	void write_net_conv(const std::string& layer_name, const caffe::LayerParameter& param){
		printf("Layer: conv %s\n",layer_name.c_str());
		const caffe::ConvolutionParameter& conv_param = param.convolution_param();
		printf("%d %d %d %d none\n",
				(int)conv_param.kernel_size()[0], 
				(int)conv_param.num_output(), 
				(int)conv_param.pad()[0], 
				(int)conv_param.stride()[0]);
	}
	void write_net_pool(const std::string& layer_name, const caffe::LayerParameter& param){
		printf("Layer: pool %s\n",layer_name.c_str());
		const caffe::PoolingParameter& pool_param = param.pooling_param();
		string method;
		switch (pool_param.pool()){
			case caffe::PoolingParameter_PoolMethod_MAX:
			method = "max";
			break;
			case caffe::PoolingParameter_PoolMethod_AVE:
			method = "avg";
			break;
			default:
			printf("should not touch here\n");
		}
		printf("%d %d %d %s\n",
			pool_param.kernel_size(),
			pool_param.pad(),
			pool_param.stride(),
			method.c_str());
	}
	void write_net_data(const std::string& layer_name, const std::string& blob_name, int n,int c,int h,int w){
		printf("Layer: data %s\n", (layer_name + "_" + blob_name).c_str());
		printf("%d %d %d %d unkonwn\n",n,c,h,w);
	}
	void write_net_relu(const std::string& layer_name, const caffe::LayerParameter& param){
		printf("Layer: activity %s\n",layer_name.c_str());
		const caffe::ReLUParameter& relu_param = param.relu_param();
		printf("relu\n");
	}
	void write_net_fc(const std::string& layer_name, const caffe::LayerParameter& param){
		printf("Layer: fc %s\n",layer_name.c_str());
		const caffe::InnerProductParameter& fc_param = param.inner_product_param();
		printf("%d none\n",(int)fc_param.num_output());
	}
	void write_net_softmaxloss(const std::string& layer_name, const caffe::LayerParameter& param){
		printf("Layer: loss %s\n",layer_name.c_str());
		printf("softmax\n");
	}
	void write_net(const std::string& net_path){
		const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
		const vector<string> layer_names = net->layer_names();
		const vector<string> blob_names = net->blob_names();
		const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();

		for(int i=0;i<layers.size();++i){
			string layer_type = layers[i]->type();
			string layer_name = layer_names[i];

			const caffe::LayerParameter& param = layers[i]->layer_param();
			if(layer_type == "Input"){
				const vector<int> top_ids = net->top_ids(i);
				for(int j=0;j<top_ids.size();++j){
					caffe::Blob<float> *b = tops[i][j];
					write_net_data(layer_name, blob_names[top_ids[j]],b->num(),b->channels(),b->height(),b->width());
				}
			}
			else if(layer_type == "Convolution") write_net_conv(layer_name, param);
			else if(layer_type == "Pooling") write_net_pool(layer_name, param);
			else if(layer_type == "ReLU") write_net_relu(layer_name, param);
			else if(layer_type == "InnerProduct") write_net_fc(layer_name, param);
			else if(layer_type == "SoftmaxWithLoss") write_net_softmaxloss(layer_name, param);
			else{
				printf("unknown layer: %s\n",layer_type.c_str());
				exit(0);
			}
		}
	}

	void show_layers(){
		const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
		const vector<string> layer_names = net->layer_names();
		const vector<string> blob_names = net->blob_names();

		for(int i=0;i<layers.size();++i){
			printf("Layers: %s %s\n",layers[i]->type(), layer_names[i].c_str());
			const vector<int> top_ids = net->top_ids(i);
			const vector<int> bottom_ids = net->bottom_ids(i);
			for(int j=0;j<bottom_ids.size();++j)
				printf("bottom Blob: %s\n",blob_names[bottom_ids[j]].c_str());
			for(int j=0;j<top_ids.size();++j)
				printf("top Blob: %s\n",blob_names[top_ids[j]].c_str());
		}
	}
private:
	std::shared_ptr<caffe::Net<float> > net;
};

void test_model(){
	string net_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_train_test.prototxt";
	string model_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_iter_5000.caffemodel.h5";
	string img_path = "/Users/worm004/Projects/net004/caffe_example/westerdam-ship-size.jpg";

	// load caffe
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TRAIN);
	net->CopyTrainedLayersFrom(model_path);
	//show_model(net);
	net->input_blobs()[0]->Reshape(1, 3, 32, 32);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data();
	float * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = 8;

	// load img
	Mat img = imread(img_path);
	resize(img,img,Size(32,32));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<32;++i){
		for(int j=0;j<32;++j){
			input[(i*32+j) + 32*32*0] = data[(i*32+j)*3+2] - 127;
			input[(i*32+j) + 32*32*1] = data[(i*32+j)*3+1] - 127;
			input[(i*32+j) + 32*32*2] = data[(i*32+j)*3+0] - 127;
		}
	}
	
	// forward
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	const caffe::Blob<float>* blob = net->Forward()[0];

	auto t2 = now();
	cout<<"forward: "<<cal_duration(t1,t2)<<" ms"<<endl;
	const float* output =  blob->cpu_data();
	for(int i=0;i<1;++i)
		printf("%f ",output[i]);
	printf("\n");
}
int main(){
	// test_model();
	
	// show layers
	string net_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_train_test.prototxt";
	string model_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_iter_5000.caffemodel.h5";
	CaffeModelParser parser;
	parser.load_caffe_model(net_path, model_path);
	//parser.show_layers();
	parser.write("a.net","a.data");
	return 0;
}
