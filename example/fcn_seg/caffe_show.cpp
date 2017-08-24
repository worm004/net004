#include <iostream>
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
void show_param_diff(caffe::Blob<float> *b, ofstream& ofile){
	const float *data = b->cpu_diff();
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

			if(params.size()>1){
				ofile << "bias: "<<params[1]->shape_string()<<endl;
				show_param(params[1].get(), ofile);
			}
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
void load_det_img(const std::shared_ptr<caffe::Net<float> >& net, const std::string& img_path, float mean_r, float mean_g, float mean_b,
		float std_r, float std_g, float std_b){

	Mat img = imread(img_path);
	resize(img,img,Size(0,0),0.5,0.5);

	int c = img.channels(), h = img.rows, w = img.cols;

	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data();

	uchar* data = (uchar*)img.data;
	for(int i=0;i<h;++i){
		for(int j=0;j<w;++j){
			input[(i*w+j) + w*h*0] = (data[(i*w+j)*3+2] - mean_r)/std_r;
			input[(i*w+j) + w*h*1] = (data[(i*w+j)*3+1] - mean_g)/std_g;
			input[(i*w+j) + w*h*2] = (data[(i*w+j)*3+0] - mean_b)/std_b;
		}
	}
}

void load_img(const std::shared_ptr<caffe::Net<float> >& net, const std::string& img_path, int label, float mean_r, float mean_g, float mean_b,
		float std_r, float std_g, float std_b){
	int c = net->input_blobs()[0]->channels(),
	    h = net->input_blobs()[0]->height(), 
	    w = net->input_blobs()[0]->width();

	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->input_blobs()[1]->Reshape(1, 1, 1, 1);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data(),
	      * input2 = net->input_blobs()[1]->mutable_cpu_data();
	input2[0] = label;

	Mat img = imread(img_path);
	resize(img,img,Size(h,w));
	uchar* data = (uchar*)img.data;
	for(int i=0;i<h;++i){
		for(int j=0;j<w;++j){
			input[(i*w+j) + w*h*0] = (data[(i*w+j)*3+2] - mean_r)/std_r;
			input[(i*w+j) + w*h*1] = (data[(i*w+j)*3+1] - mean_g)/std_g;
			input[(i*w+j) + w*h*2] = (data[(i*w+j)*3+0] - mean_b)/std_b;
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
		if((layers[i]->type() == string("BatchNorm")) || (layers[i]->type() == string("Scale")) || (layers[i]->type() == string("ReLU")))
			continue;

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
		const vector<boost::shared_ptr<caffe::Blob<float> > >& params = layers[i]->blobs();
		string layer_type = layers[i]->type();

		ofile<<layer_type<<" "<<bottoms[i].size()<<" "<<tops[i].size()<<" "<<params.size()<<endl;

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

		if((layer_type == "Convolution") || (layer_type == "InnerProduct")){
			ofile << "weight diff: "<<params[0]->shape_string()<<endl;
			show_param_diff(params[0].get(), ofile);

			ofile << "bias diff: "<<params[1]->shape_string()<<endl;
			show_param_diff(params[1].get(), ofile);
		}
		else{
			for(int j=0;j<params.size();++j){
				ofile << "param diff: "<<params[j]->shape_string()<<endl;
				show_param_diff(params[j].get(), ofile);
			}
		}
	}

	ofile.close();
}
int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	// yolo
	float mean_b = 122.67892, mean_g = 116.66877, mean_r = 104.00699;
	float std_r = 1, std_g = 1, std_b = 1;
	string net_path = "../caffe_models/segmentation/fcn8s.prototxt",
		model_path = "../caffe_models/segmentation/fcn8s-atonce-pascal.caffemodel";
	string model_text_path = "model.txt",
	       forward_text_path = "forward.txt",
	       backward_text_path = "backward.txt";

  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	net = make_shared<caffe::Net<float>> (net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(model_path);
	//show_model(net, model_text_path);

	//string img_path = "../imgs/westerdam-ship-size.jpg";// for classification
	string img_path = "../imgs/person.jpg";//for detection
	//load_img(net,img_path,label,mean_r,mean_g,mean_b,std_r,std_g,std_b);//for classification
	load_det_img(net,img_path,mean_r,mean_g,mean_b,std_r,std_g,std_b);// for detection
	net->Forward();
	show_forward(net, forward_text_path);
	//net->Backward();
	//show_backward(net, backward_text_path);

	return 0;
}
