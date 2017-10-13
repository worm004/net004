#include "stdlib.h"
#include "DisplayRun.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
DisplayRun::DisplayRun(){}
DisplayRun::DisplayRun(const JsonValue& j):Run(j){
	if(omit) return;
	const auto& a = j.jobj.at("attrs").jobj.at("layers").jarray;
	for(const auto& i: a) layers.push_back(i.jv.s);
}
void DisplayRun::show()const{
	if(omit) return;
	Run::show();
	printf("  (layers) ");
	for(int i=0;i<layers.size();++i){
		if(i != 0) printf(", ");
		printf("%s",layers[i].c_str());
	}
	printf("\n");
}
void DisplayRun::check(const Net004& net)const{
	if(omit) return;
	Run::check(net);
	for(const auto& i: layers)
		if (net.ls.n2i.find(i) == net.ls.n2i.end()){
			printf("cannot find layer: %s\n",i.c_str());
			exit(0);
		}
}
void DisplayRun::operator()(Net004& net, int cur){
	if(omit) return;
	if(cur%iter_interval != 0) return;
	if(0)
	{
		Layer *l = net.ls["conv1"];
		const Blob& b = l->params["weight"];
		float *data = b.data,  max_v = -1e10, min_v = 1e10;
		int nchw = b.nchw();
		for(int i=0;i<nchw;++i){
			max_v = max(max_v, data[i]);
			min_v = min(min_v, data[i]);
		}

		int unit = 10;
		cv::Mat show = cv::Mat::zeros(b.c*unit*b.h, b.n*unit*b.w,CV_8UC1);
		for(int i=0;i<b.n;++i){
			for(int j=0;j<b.c;++j){
				for(int ii = 0;ii<b.h;++ii){
					for(int jj = 0;jj<b.w;++jj){
						float v = data[jj + ii*b.w + j * b.hw() + i * b.chw()];
						float nv = (v - min_v)/(max_v - min_v);
						rectangle(show, Rect(i*unit*b.w+unit*jj,j*unit*b.h+unit*ii,unit,unit),Scalar(nv*255.0f+0.5),-1);
					}
				}
			}
		}
		imshow("conv1",show);
	}
	if(0)
	{
		Layer *l = net.ls["conv2"];
		const Blob& b = l->params["weight"];
		float *data = b.data,  max_v = -1e10, min_v = 1e10;
		int nchw = b.nchw();
		for(int i=0;i<nchw;++i){
			max_v = max(max_v, data[i]);
			min_v = min(min_v, data[i]);
		}

		int unit = 10;
		cv::Mat show = cv::Mat::zeros(b.c*unit*b.h, b.n*unit*b.w,CV_8UC1);
		for(int i=0;i<b.n;++i){
			for(int j=0;j<b.c;++j){
				for(int ii = 0;ii<b.h;++ii){
					for(int jj = 0;jj<b.w;++jj){
						float v = data[jj + ii*b.w + j * b.hw() + i * b.chw()];
						float nv = (v - min_v)/(max_v - min_v);
						rectangle(show, Rect(i*unit*b.w+unit*jj,j*unit*b.h+unit*ii,unit,unit),Scalar(nv*255.0f+0.5),-1);
					}
				}
			}
		}
		imshow("conv2",show);
	}
	if(0)
	{
		Layer *l = net.ls["conv3"];
		const Blob& b = l->params["weight"];
		float *data = b.data,  max_v = -1e10, min_v = 1e10;
		int nchw = b.nchw();
		for(int i=0;i<nchw;++i){
			max_v = max(max_v, data[i]);
			min_v = min(min_v, data[i]);
		}

		int unit = 10;
		cv::Mat show = cv::Mat::zeros(b.c*unit*b.h, b.n*unit*b.w,CV_8UC1);
		for(int i=0;i<b.n;++i){
			for(int j=0;j<b.c;++j){
				for(int ii = 0;ii<b.h;++ii){
					for(int jj = 0;jj<b.w;++jj){
						float v = data[jj + ii*b.w + j * b.hw() + i * b.chw()];
						float nv = (v - min_v)/(max_v - min_v);
						rectangle(show, Rect(i*unit*b.w+unit*jj,j*unit*b.h+unit*ii,unit,unit),Scalar(nv*255.0f+0.5),-1);
					}
				}
			}
		}
		imshow("conv3",show);
	}

	//waitKey(1);
	//for(auto ln:layers){
	//	Layer* layer = net[ln];
	//	for(int i=0;i<layer->outputs.size();++i){
	//		int chw = layer->outputs[i].chw();
	//		float *data = layer->outputs[i].data;
	//		int batch_size = layer->outputs[i].n;
	//		for(int b=0;b<batch_size;++b){
	//			printf("[iter %07d] [display] [layer %s] [output %d] [batch %d]:",cur,ln.c_str(),i,b);
	//			for(int j=0;j<chw;++j)
	//				printf(" %g",data[j+b*chw]);
	//			printf("\n");
	//		}
	//	}
	//}
}
