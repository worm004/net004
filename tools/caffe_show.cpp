#include <string>
#include <fstream>
#include <vector>
#include "caffe/caffe.hpp"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
using namespace std;
using namespace cv;

void show_param(caffe::Blob<float> *b, ofstream& ofile){
	const float *data = b->cpu_data();
	int total = b->num() * b->channels() * b->height() * b->width();
	for(int i=0;i<total;++i)
		ofile << " "<<data[i];
	ofile << endl;
}
void show_model(const std::shared_ptr<caffe::Net<float> >& net, const std::string& path ){
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	ofstream ofile(path);
	for(int i=0;i<layers.size();++i){
		const vector<boost::shared_ptr<caffe::Blob<float> > >& params = layers[i]->blobs();
		string layer_type = layers[i]->type();
		ofile << layer_type <<" "<<params.size()<<endl;
		if((layer_type == "Convolution") || (layer_type == "InnerProduct")){
			ofile << "weight: "<<params[0]->shape_string()<<endl;
			show_param(params[0].get(), ofile);

			ofile << "bias: "<<params[1]->shape_string()<<endl;
			show_param(params[1].get(), ofile);
		}
		else{
			for(int j=0;j<params.size();++j){
				ofile << "param: "<<params[j]->shape_string()<<endl;
				show_param(params[j].get(), ofile);
			}
		}
	}
	ofile.close();
}
void load_img(const std::shared_ptr<caffe::Net<float> >& net, const std::string& img_path, int label){
	net->input_blobs()[0]->Reshape(1, 3, 32, 32);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data();
	float * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = label;

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
}
void show_blob_diff(caffe::Blob<float> *b, ofstream& ofile){
	const float *data = b->cpu_diff();
	int total = b->num() * b->channels() * b->height() * b->width();
	for(int i=0;i<total;++i)
		ofile<<" "<<data[i];
	ofile<<endl;
}
void show_blob(caffe::Blob<float> *b, ofstream& ofile){
	const float *data = b->cpu_data();
	int total = b->num() * b->channels() * b->height() * b->width();
	for(int i=0;i<total;++i)
		ofile<<" "<<data[i];
	ofile<<endl;

	//int num = b->num(), channel = b->channels(), height = b->height(), width = b->width();
	//for(int n=0;n<num;++n){
	//	//printf("batch %d/%d\n",n+1,num);
	//	for(int c=0;c<channel;++c){
	//		//printf("channel %d/%d\n",c+1,channel);
	//		for(int h=0;h<height;++h){
	//			for(int w=0;w<width;++w){
	//				printf(" %f",data[(h*width+w) + height*width*c + height*width*channel*n]);
	//			}
	//		}
	//	}
	//}

}
void show_forward(const std::shared_ptr<caffe::Net<float> >& net, const std::string& path){
	ofstream ofile(path);
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<vector<caffe::Blob<float> *> >& bottoms = net->bottom_vecs();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();

	for(int i=0;i<layers.size();++i){
		ofile<<layers[i]->type()<<" "<<bottoms[i].size()<<" "<<tops[i].size()<<endl;
		//printf("%s %lu %lu\n",layers[i]->type(), bottoms[i].size(), tops[i].size());
		for(int j=0;j<bottoms[i].size();++j){
			caffe::Blob<float> *b = bottoms[i][j];
			ofile<<"bottoms ("<<j<<"): "<<b->shape_string()<<endl;
			show_blob(b,ofile);
		}
		for(int j=0;j<tops[i].size();++j){
			const float *data = tops[i][j]->cpu_data();
			caffe::Blob<float> *b = tops[i][j];
			ofile<<"tops ("<<j<<"): "<<b->shape_string()<<endl;
			show_blob(b,ofile);
		}
	}

	ofile.close();
}
void show_backward(const std::shared_ptr<caffe::Net<float> >&net, const std::string& path){
	ofstream ofile(path);
	const vector<boost::shared_ptr<caffe::Layer<float> >> layers = net->layers();
	const vector<vector<caffe::Blob<float> *> >& bottoms = net->bottom_vecs();
	const vector<vector<caffe::Blob<float> *> >& tops = net->top_vecs();

	for(int i=0;i<layers.size();++i){
		ofile<<layers[i]->type()<<" "<<bottoms[i].size()<<" "<<tops[i].size()<<endl;
		for(int j=0;j<bottoms[i].size();++j){
			caffe::Blob<float> *b = bottoms[i][j];
			ofile<<"bottoms diff("<<j<<"): "<<b->shape_string()<<endl;
			show_blob_diff(b,ofile);
		}
		for(int j=0;j<tops[i].size();++j){
			const float *data = tops[i][j]->cpu_data();
			caffe::Blob<float> *b = tops[i][j];
			ofile<<"tops diff("<<j<<"): "<<b->shape_string()<<endl;
			show_blob_diff(b,ofile);
		}
	}

	ofile.close();
}
int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	string net_path = "../caffe_models/cifar10_quick_train_test.prototxt",
	       model_path = "../caffe_models/cifar10_quick_iter_5000.caffemodel.h5",
	       model_text_path = "model.txt",
	       forward_text_path = "forward.txt",
	       backward_text_path = "backward.txt";
	
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	show_model(net, model_text_path);

	string img_path = "../imgs/westerdam-ship-size.jpg";
	int label = 8;
	load_img(net,img_path,label);
	net->Forward();
	show_forward(net, forward_text_path);
	net->Backward();
	show_backward(net, backward_text_path);

	return 0;
}
