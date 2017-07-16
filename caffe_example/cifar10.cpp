#include "stdio.h"
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
void show_param(caffe::Blob<float> *b){
	const float *data = b->cpu_data();
	int num = b->num(), channel = b->channels(), height = b->height(), width = b->width();
	for(int n=0;n<num;++n){
		printf("filter %d/%d\n",n+1,num);
		for(int c=0;c<channel;++c){
			if((height >1) || (width > 1)) printf("channel %d/%d\n",c+1,channel);
			for(int h=0;h<height;++h){
				for(int w=0;w<width;++w){
					printf(" %.2f ",data[(h*width+w) + height*width*c] + height*width*channel*n);
				}
			}
			if((height >1) || (width > 1)) printf("\n");
		}
		printf("\n");

	}
}
void show_blob(caffe::Blob<float> *b){
	const float *data = b->cpu_data();
	int num = b->num(), channel = b->channels(), height = b->height(), width = b->width();
	for(int n=0;n<num;++n){
		printf("batch %d/%d\n",n+1,num);
		for(int c=0;c<channel;++c){
			printf("channel %d/%d\n",c+1,channel);
			for(int h=0;h<height;++h){
				for(int w=0;w<width;++w){
					printf(" %.2f ",data[(h*width+w) + height*width*c] + height*width*channel*n);
				}
			}
			printf("\n");
		}
		printf("\n");
	}

}
void show_data_flow(const std::shared_ptr<caffe::Net<float> >& net){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<vector<caffe::Blob<float> *> >& bottoms = net->bottom_vecs();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();

	for(int i=0;i<layers.size();++i){
		printf("%s %lu %lu\n",layers[i]->type(), bottoms[i].size(), tops[i].size());

		for(int j=0;j<bottoms[i].size();++j){
			caffe::Blob<float> *b = bottoms[i][j];
			printf("bottoms(%d): %s\n",j,b->shape_string().c_str());
			show_blob(b);
		}

		for(int j=0;j<tops[i].size();++j){
			const float *data = tops[i][j]->cpu_data();
			caffe::Blob<float> *b = tops[i][j];
			printf("tops(%d): %s\n",j,tops[i][j]->shape_string().c_str());
			show_blob(b);
		}
	}
	getchar();
}
void show_model(const std::shared_ptr<caffe::Net<float> >& net){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	for(int i=0;i<layers.size();++i){
		const vector<boost::shared_ptr<caffe::Blob<float> > >& params = layers[i]->blobs();
		printf("%s %lu\n",layers[i]->type(), params.size());
		for(int j=0;j<params.size();++j){
			printf("param %d: %s\n",j, params[j]->shape_string().c_str());
			show_param(params[j].get());
		}
	}

	//const vector<boost::shared_ptr<caffe::Blob<float> > >& params = net->params();
	//const vector<string>& names = net->param_display_names();
	//for(int i=0;i<params.size();++i){
	//	printf("%d %s %s\n",i,names[i].c_str(), params[i]->shape_string().c_str());
	//}
}
int main(){
	vector<string> classes = { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

	string net_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_train_test.prototxt";
	string model_path = "/Users/worm004/Projects/net004/caffe_example/cifar10_quick_iter_5000.caffemodel.h5";
	//string img_path = "/Users/worm004/Projects/net004/caffe_example/tabby-cat-names.jpg";
	string img_path = "/Users/worm004/Projects/net004/caffe_example/westerdam-ship-size.jpg";

	// load caffe
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	net->input_blobs()[0]->Reshape(1, 3, 32, 32);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data();

	// load img
	Mat img = imread(img_path);
	resize(img,img,Size(32,32));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<32;++i){
		for(int j=0;j<32;++j){
			input[(i*32+j) + 32*32*0] = data[(i*32+j)*3+2] - 127;
			input[(i*32+j) + 32*32*1] = data[(i*32+j)*3+1] - 127;
			input[(i*32+j) + 32*32*2] = data[(i*32+j)*3+0] - 127;
			//printf("[%.1f",input[(i*32+j) + 32*32*0]);
			//printf(" %.1f",input[(i*32+j) + 32*32*1]);
			//printf(" %.1f] ",input[(i*32+j) + 32*32*2]);
		}
		//printf("\n");
	}
	
	// forward
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	const caffe::Blob<float>* blob = net->Forward()[0];
	show_model(net);
	//show_data_flow(net);
	const float* output =  blob->cpu_data();
	for(int i=0;i<10;++i)
		printf("%s %.2f\n",classes[i].c_str(),output[i]);

	return 0;
}
