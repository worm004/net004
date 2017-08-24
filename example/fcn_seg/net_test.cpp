#include <map>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "caffe/caffe.hpp"
#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace cv;
using namespace std;
struct TestParameter{
	string caffe_net_path = "../caffe_models/segmentation/fcn8s.prototxt",
		caffe_model_path = "../caffe_models/segmentation/fcn8s-atonce-pascal.caffemodel",
		net004_net_path = "../models/segmentation/fcn8s.net004.net",
		net004_model_path = "../models/segmentation/fcn8s.net004.data";
	float mean_r = 104.00699, mean_g = 116.66877, mean_b = 122.67892;
	float std_r = 1.0, std_g = 1.0, std_b = 1.0;
	int cnum = 20;
};
void net004_forward(const cv::Mat& img, const TestParameter& param, bool show, float** ret){
	if(show) printf("net004 forwarding ...\n");
	Net004 net;
	Parser parser;
	parser.set_input_size("data",1,img.channels(),img.rows,img.cols);
	auto t1 = now();
	parser.read(param.net004_net_path, param.net004_model_path, &net);
	auto t2 = now();

	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	Layers & ls = net.ls;
	int c = img.channels(), h = img.rows, w = img.cols;
	float *idata = (float*)(DataLayer*)ls["data"]->outputs[0].data, *data = (float*)img.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j)
	for(int k=0;k<c;++k)
		idata[(i*w+j) + h*w*k] = data[(i*w+j)*c+k]; 

	t1 = now();
	net.forward();
	t2 = now();

	if(show) cout<<"forward: "<<cal_duration(t1,t2)<<endl;

//
//	Connections &cs = net.cs;
//
//	float xyscale = float(rimg.cols) / float(rimg.rows);
//	parser_yolov1(ls[cs.sorted_cs.back()]->outputs[0].data, param.cnum, rs, xyscale, param.threshold);
//
//	vector<string> names(param.cnum);
//	ifstream file(param.list_path);
//	for(int i=0;i<param.cnum;++i)
//		file >> names[i];
//
//	for(int i=0;i<(rs.size())&&show;++i){
//		printf("%s (%f) %f %f %f %f\n",names[int(rs[i][5])].c_str(),rs[i][4],rs[i][0],rs[i][1],rs[i][2],rs[i][3]);
//	}
//
}

void process_image(const std::string& path, const TestParameter& param, cv::Mat& ret){
	Mat img = imread(path);
	resize(img,img,Size(),0.5,0.5);
	ret = cv::Mat(img.size(),CV_32FC3);

	int h = ret.rows, w = ret.cols, c = ret.channels();
	float* odata = (float*) ret.data;
	uchar* idata = (uchar*) img.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		int index = (i*w+j)*c;
		odata[index + 0] = idata[index + 2] - param.mean_r;
		odata[index + 1] = idata[index + 1] - param.mean_g;
		odata[index + 2] = idata[index + 0] - param.mean_b;
	}
}
void show_seg(const cv::Mat& img, const float *scores, const TestParameter& param, float T){
	vector<string> names(param.cnum+1);
	names[0] = "background";
	ifstream file("../caffe_models/detection/voc.list" );
	for(int i=1;i<=20;++i)
		file >> names[i];

	int h = img.rows, w = img.cols, c = img.channels();
	for(int i=0;i<param.cnum;++i){
		bool flag = false;
		Mat show = img.clone();
		uchar *data = (uchar*)show.data;
		for(int y=0, index = 0;y<h;++y)
		for(int x=0;x<w;++x,++index)
			if(scores[index + i*w*h] > T){
				data[index*c+2] =255;
				flag = true;
			}
		if(!flag) continue;
		imshow(names[i].c_str(),show);
	}
	waitKey();
}
void caffe_forward(const cv::Mat& img, const TestParameter& param, bool show, float** ret){
	if(show) printf("caffe forwarding ...\n");
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (param.caffe_net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(param.caffe_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	int c = img.channels(), h = img.rows, w = img.cols;
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->Reshape();

	float * idata = net->input_blobs()[0]->mutable_cpu_data(), 
	      * data = (float*)img.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j)
	for(int k=0;k<3;++k)
		idata[(i*w+j) + h*w*k] = data[(i*w+j)*3+k]; 

	net->Forward()[0];
	t1 = now();
	net->Forward()[0];
	t2 = now();
	if(show) cout<<"forward: "<<cal_duration(t1,t2)<<endl;
	memcpy(*ret,net->blob_by_name("sm")->cpu_data(),sizeof(float)*h*w*(param.cnum+1));
}
int main(int argc, char** argv){
	if(argc !=2){
		printf("./net_test_fcn_seg 0/1\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	TestParameter param;
	bool show = atoi(argv[1]);
	string img_path = "../imgs/person.jpg";

	printf("[TEST] [forwrad] %s\n","fcn_seg");

	Mat img;
	process_image(img_path, param, img);
	float *caffe_scores = new float[img.cols*img.rows*(param.cnum+1)];
	//caffe_forward(img, param, show, &caffe_scores);
	//Mat img2 = imread(img_path);
	//resize(img2,img2,img.size());
	//show_seg(img2, caffe_scores,param,0.4);
	
	float *net004_scores = new float[img.cols*img.rows*(param.cnum+1)];
	net004_forward(img, param, show, &net004_scores);

}
