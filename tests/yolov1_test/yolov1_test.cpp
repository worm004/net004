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
	TestParameter(){}
	TestParameter(const std::string& cmp, const std::string& cnp, 
			const std::string& nmp, const std::string& nnp, const std::string& lp,
			float mr, float mg, float mb, float sr, float sg, float sb, float t, int cn){
		caffe_model_path = cmp;
		caffe_net_path = cnp;
		net004_model_path = nmp;
		net004_net_path = nnp;
		mean_r = mr;
		mean_g = mg;
		mean_b = mb;
		std_r = sr;
		std_g = sg;
		std_b = sb;
		threshold = t;
		cnum = cn;
		list_path =lp;
	}
	
	string caffe_model_path,
		caffe_net_path,
		net004_model_path,
		net004_net_path;
	string list_path;
	float mean_r, mean_g, mean_b, std_r, std_g, std_b;
	float threshold;
	int cnum;
};

void parser_yolov1(const float* data, int cnum,vector<vector<float> >& rects, float xyscale, float threshold){
	int class_number = cnum, slide = 7, obj_number = 2;
	int widthstep = slide * slide,
		offset_scale = widthstep * obj_number,
		offset_xywh = widthstep * 4;

	std::vector<float> rs(4);
	for(int i=0, loc = 0;i<slide;++i)
	for(int j=0;j<slide;++j, ++loc){
		const float * pscore = data + loc;
		float score = 0.0f;
		int c = -1;
		for(int k=0;k<class_number;++k){
			if(*pscore > score){
				score = *pscore;
				c = k;
			}
			pscore += widthstep;
		}
		const float * pscale = pscore,
		      * px = pscale + offset_scale, 
		      * py = px + widthstep, 
		      * pw = py + widthstep, 
		      * ph = pw + widthstep;

		for(int k=0;k<obj_number;++k){
			float real_score = *pscale * score;
			if(real_score < threshold) continue;

			float w = *pw * *pw,
			      h = *ph * *ph,
			      x = (*px+j)/slide - w/2.0f,
			      y = (*py+i)/slide - h/2.0f;

			if(xyscale > 1.0f){
				rs[0]=x; rs[1]= (y - 0.5f*(1.0f-1.0f/xyscale)) * xyscale;
				rs[2]=w; rs[3]= h *xyscale;
			}
			else {
				rs[0]=(x- 0.5f*(1.0f-1.0f*xyscale))*xyscale; rs[1]=y;
				rs[2]=w/xyscale; rs[3]=h;
			}
			rects.push_back({rs[0],rs[1],rs[2],rs[3],real_score,float(c)});

			pscale += widthstep;
			px += offset_xywh;
			py += offset_xywh;
			pw += offset_xywh;
			ph += offset_xywh;
		}
	}
}

void net004_forward(const std::string& img_path, const TestParameter& param, bool show, vector<vector<float> >& rs){
	if(show) printf("net004 forwarding ...\n");
	Net004 net;
	Parser parser;
	auto t1 = now();
	parser.read(param.net004_net_path, param.net004_model_path, &net);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;
	Layers & ls = net.ls;
	DataLayer* l = (DataLayer*)ls["data"];
	Mat img = imread(img_path);
	Mat rimg;
	resize(img,rimg,Size(l->outputs[0].h, l->outputs[0].w));
	l->add_image((uchar*)rimg.data,0, param.mean_r, param.mean_g, param.mean_b,param.std_r,param.std_g,param.std_b);

	t1 = now();
	net.forward();
	t2 = now();
	if(show){
		cout<<"forward: "<<cal_duration(t1,t2)<<endl;
		//net.show();
	}

	Connections &cs = net.cs;

	float xyscale = float(rimg.cols) / float(rimg.rows);
	parser_yolov1(ls[cs.sorted_cs.back()]->outputs[0].data, param.cnum, rs, xyscale, param.threshold);

	vector<string> names(param.cnum);
	ifstream file(param.list_path);
	for(int i=0;i<param.cnum;++i)
		file >> names[i];

	for(int i=0;i<(rs.size())&&show;++i){
		printf("%s (%f) %f %f %f %f\n",names[int(rs[i][5])].c_str(),rs[i][4],rs[i][0],rs[i][1],rs[i][2],rs[i][3]);
	}

	//for(int i=0;i<rs.size();++i){
	//	int x = rs[i][0] * img.cols, y = rs[i][1] * img.rows, w = rs[i][2] * img.cols, h = rs[i][3] * img.rows;
	//	for(int j=0;j<6;++j)
	//		printf("%g ",rs[i][j]);
	//	printf("\n");
	//	rectangle(img,Rect(x,y,w,h),Scalar(0,0,255),2);
	//}
	//imshow("img",img);
	//waitKey();

}
void caffe_forward(const std::string& img_path, const TestParameter& param, bool show, vector<vector<float> >& rs){

	if(show) printf("caffe forwarding ...\n");
  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (param.caffe_net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(param.caffe_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	int c = net->input_blobs()[0]->channels(),
	    h = net->input_blobs()[0]->height(), 
	    w = net->input_blobs()[0]->width();
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->Reshape();

	Mat img = imread(img_path);
	Mat rimg;
	resize(img,rimg,Size(h,w));
	uchar* data = (uchar*)rimg.data;
	float * input = net->input_blobs()[0]->mutable_cpu_data();
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		input[(i*w+j) + h*w*0] = (data[(i*w+j)*3+2] - param.mean_r)/param.std_r;
		input[(i*w+j) + h*w*1] = (data[(i*w+j)*3+1] - param.mean_g)/param.std_g;
		input[(i*w+j) + h*w*2] = (data[(i*w+j)*3+0] - param.mean_b)/param.std_b;
	}
	net->Forward()[0];
	t1 = now();
	const caffe::Blob<float>* blob = net->Forward()[0];
	t2 = now();
	if(show) cout<<"forward: "<<cal_duration(t1,t2)<<endl;

	float xyscale = float(rimg.cols) / float(rimg.rows);
	parser_yolov1(blob->cpu_data(),param.cnum,rs,xyscale, param.threshold);
	
	vector<string> names(param.cnum);
	ifstream file(param.list_path);
	for(int i=0;i<param.cnum;++i)
		file >> names[i];

	for(int i=0;i<(rs.size())&&show;++i){
		printf("%s (%f) %f %f %f %f\n",names[int(rs[i][5])].c_str(),rs[i][4],rs[i][0],rs[i][1],rs[i][2],rs[i][3]);
	}

	//for(int i=0;i<rs.size();++i){
	//	int x = rs[i][0] * img.cols, y = rs[i][1] * img.rows, w = rs[i][2] * img.cols, h = rs[i][3] * img.rows;
	//	for(int j=0;j<6;++j)
	//		printf("%g ",rs[i][j]);
	//	printf("\n");
	//	rectangle(img,Rect(x,y,w,h),Scalar(0,0,255),2);
	//}
	//imshow("img",img);
	//waitKey();
}
int main(int argc, char** argv){
	if(argc !=2){
		printf("./yolov1_test 0/1\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	map<string, TestParameter> maps;

	maps["yolov1"] = TestParameter(
	       "../caffe_models/detection/gnet_yolo_iter_32000.caffemodel",
	       "../caffe_models/detection/gnet_deploy.prototxt",
	       "../models/detection/yolov1.net004.data",
	       "../models/detection/yolov1.net004.net",
	       "../caffe_models/detection/voc.list",
	       128, 128, 128, 1, 1, 1, 0.1, 20
	);
	bool show = atoi(argv[1]);
	string img_path = "../imgs/person.jpg";

	if(maps.find("yolov1")!= maps.end()){
		printf("[TEST] [forwrad] %s\n","yolov1");
		vector<vector<float> > caffe_rs, rs;
		caffe_forward(img_path, maps["yolov1"], show, caffe_rs);
		net004_forward(img_path,maps["yolov1"], show, rs);

		if(rs.size()!=caffe_rs.size()){
			printf("caffe rs: %lu\nnet004 rs: %lu\n",caffe_rs.size(),rs.size());
			printf("[TEST] [result] %s\n","\x1B[31mfailed"); // red failed
		}
		else {
			bool is_same = true;
			for(int i=0;(i<rs.size()) && is_same;++i)
			for(int j=0;(j<5)&&is_same;++j)
				if(abs(caffe_rs[i][j] - rs[i][j])>1e-4)
					is_same = false;
			printf("[TEST] [result] %s\n",is_same?"sucessful":"\x1B[31mfailed"); // red failed
		}
	}

	return 0;
}
