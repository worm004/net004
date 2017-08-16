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
#include "stdlib.h"
#include "Python.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace cv;
using namespace std;
using namespace caffe;

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

void parser_faster_rcnn(const float* data, int cnum,vector<vector<float> >& rects, float xyscale, float threshold){
}

void preprocess_img(const cv::Mat& src, cv::Mat& des, float &scale, int target_scale, int max_size, float r, float g, float b){
	des = Mat::zeros(src.size(),CV_32FC3);
	float *odata = (float*)des.data;
	uchar *idata = (uchar*)src.data;
	int n = src.rows * src.cols * src.channels();

	for(int i=0;i<n;i+=3){
		odata[i] =   float(idata[i]) - b;
		odata[i+1] = float(idata[i+1]) - g;
		odata[i+2] = float(idata[i+2]) - r;
	}

	int min_s = min(src.cols,src.rows), max_s = max(src.cols,src.rows);
	float s = float(target_scale)/float(min_s);
	if(int(s * max_s+0.5) > max_size) s = float(max_size) / float(max_s);

	resize(des,des,Size(),s,s,CV_INTER_LINEAR);
	scale = s;
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
	parser_faster_rcnn(ls[cs.sorted_cs.back()]->outputs[0].data, param.cnum, rs, xyscale, param.threshold);

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
void bbox_transform_inv(const std::vector<vector<float> >& boxes, const std::vector<vector<float> >& deltas, std::vector<vector<float> >& pred_boxes){
	int n = deltas.size(), c = deltas[0].size();
	pred_boxes.resize(n,vector<float>(c));
	for(int i=0;i<n;++i){
		float width = boxes[i][2] - boxes[i][0] + 1.0f;
		float height = boxes[i][3] - boxes[i][1] + 1.0f;
		float cx = boxes[i][0] + 0.5*width;
		float cy = boxes[i][1] + 0.5*height;
		//printf("%f %f %f %f\n",width,height,cx,cy);

		for(int j=0;j<c;j+=4){
			float dx = deltas[i][j];
			float dy = deltas[i][j+1];
			float dw = deltas[i][j+2];
			float dh = deltas[i][j+3];
			//printf("%f %f %f %f",dx,dy,dw,dh);

			float pred_cx = dx * width + cx;
			float pred_cy = dy * height + cy;
			float pred_w = exp(dw) * width;
			float pred_h = exp(dh) * height;
			//printf("%f %f %f %f",pred_cx, pred_cy, pred_w, pred_h);

			pred_boxes[i][j] =  pred_cx - 0.5*pred_w;
			pred_boxes[i][j+1] = pred_cy - 0.5*pred_h;
			pred_boxes[i][j+2] = pred_cx + 0.5*pred_w;
			pred_boxes[i][j+3] = pred_cy + 0.5*pred_h;

		}
	}
}
void caffe_forward(const std::string& img_path, const TestParameter& param, bool show, vector<vector<float> >& rs){

	if(show) printf("caffe forwarding ...\n");

	Mat img = imread(img_path);
	Mat rimg;
	float scale;
	preprocess_img(img,rimg,scale,600,1000,param.mean_r,param.mean_g,param.mean_b);
	uchar *data = (uchar*)img.data;
	//for(int i=0;i<img.cols*img.rows*img.channels();++i)
	//	printf("%d ",data[i]);
	//printf("\n");
	//for(int i=0;i<rimg.cols*rimg.rows*rimg.channels();++i)
	//	printf("%g ",fdata[i]);
	//printf("\n");

  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (param.caffe_net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(param.caffe_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	int c = net->input_blobs()[0]->channels(), h = rimg.rows, w = rimg.cols;
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->Reshape();
	float * input = net->input_blobs()[0]->mutable_cpu_data(), 
		* input2 = net->input_blobs()[1]->mutable_cpu_data();

	float* fdata = (float*)rimg.data;
	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j){
		input[(i*w+j) + h*w*2] = fdata[(i*w+j)*3+2]; 
		input[(i*w+j) + h*w*1] = fdata[(i*w+j)*3+1];
		input[(i*w+j) + h*w*0] = fdata[(i*w+j)*3+0];
	}
	input2[0] = rimg.rows;
	input2[1] = rimg.cols;
	input2[2] = scale;

	float loss;
	net->ForwardPrefilled(&loss);
	t1 = now();
	const caffe::Blob<float>* blob = net->ForwardPrefilled(&loss)[0];
	t2 = now();
	if(show) cout<<"forward: "<<cal_duration(t1,t2)<<endl;

	const caffe::shared_ptr<caffe::Blob<float> > roi_blob = net->blob_by_name("rois");

	// rois
	const float * roi_data = roi_blob->cpu_data();
	vector<vector<float> > boxes(roi_blob->num(),vector<float>(4));
	for(int i=0;i<roi_blob->num();++i){
		boxes[i][0] = roi_data[i*roi_blob->channels()+1]/scale;
		boxes[i][1] = roi_data[i*roi_blob->channels()+2]/scale;
		boxes[i][2] = roi_data[i*roi_blob->channels()+3]/scale;
		boxes[i][3] = roi_data[i*roi_blob->channels()+4]/scale;
	}
	//for(int i=0;i<boxes.size();++i)
	//	for(int j=0;j<4;++j)
	//		printf("%g ",boxes[i][j]);
	//printf("\n");

	// bbox_pred
	const caffe::shared_ptr<caffe::Blob<float> > bbox_pred_blob = net->blob_by_name("bbox_pred");
	vector<vector<float> > pred_boxes, box_deltas(bbox_pred_blob->num(), vector<float>(bbox_pred_blob->channels()));
	const float * bbox_pred_data = bbox_pred_blob->cpu_data();
	for(int i=0;i<bbox_pred_blob->num();++i){
		for(int j=0;j<bbox_pred_blob->channels();++j){
			box_deltas[i][j] = bbox_pred_data[i*bbox_pred_blob->channels()+j];
			//printf("%g ",box_deltas[i][j]);
		}
	}
	//printf("\n");
	bbox_transform_inv(boxes, box_deltas, pred_boxes);
	for(int i=0;i<pred_boxes.size();++i){
		for(int j=0;j<pred_boxes[i].size();j+=4){
			pred_boxes[i][j] =   max(pred_boxes[i][j],0.0f);
			pred_boxes[i][j+1] = max(pred_boxes[i][j+1],0.0f);
			pred_boxes[i][j+2] = min(pred_boxes[i][j+2],float(img.cols-1));
			pred_boxes[i][j+3] = min(pred_boxes[i][j+3],float(img.rows-1));
		}
	}
	//for(int i=0;i<pred_boxes.size();++i)
	//	for(int j=0;j<pred_boxes[0].size();++j){
	//		printf("%g ",pred_boxes[i][j]);
	//	}
	//printf("\n");

	// scores
	const caffe::shared_ptr<caffe::Blob<float> > score_blob = net->blob_by_name("cls_prob");
	const float * score_data = score_blob->cpu_data();
	vector<vector<float>> scores(score_blob->num(),vector<float>(score_blob->channels()));

	for(int i=0;i<scores.size();++i)
		for(int j=0;j<scores[i].size();++j){
			scores[i][j] = score_data[i*score_blob->channels()+j];
			//printf("%g ",scores[i][j]);
		}
	//parser_faster_rcnn(blob->cpu_data(), param.cnum, rs, xyscale, param.threshold);
	
	vector<string> names(param.cnum);
	ifstream file(param.list_path);
	for(int i=0;i<param.cnum;++i)
		file >> names[i];

	float CONF_THRESH = 0.8;
	float NMS_THRESH = 0.3;
	for(int i=0;i<20;++i){
		int n = scores.size();
		vector<vector<float> > dets;
		for(int j=0;j<n;++j){
			if(scores[j][i+1] < CONF_THRESH) continue;
			dets.push_back({
				pred_boxes[j][(i+1)*4],
				pred_boxes[j][(i+1)*4+1],
				pred_boxes[j][(i+1)*4+2],
				pred_boxes[j][(i+1)*4+3],
				scores[j][i+1]
				});
			printf("%d %f %f %f %f %f\n",i,
				pred_boxes[j][(i+1)*4],
				pred_boxes[j][(i+1)*4+1],
				pred_boxes[j][(i+1)*4+2],
				pred_boxes[j][(i+1)*4+3],
				scores[j][i+1]
			);
		}
		//Mat show = img.clone();
		//for(int j=0;j<dets.size();++j){
		//	rectangle(show,Rect(dets[j][0],dets[j][1],dets[j][2]-dets[j][0],dets[j][3]-dets[j][1]),Scalar(0,0,255),2);
		//}
		//imshow(names[i].c_str(),show);
		//waitKey();
	}

	

	//for(int i=0;i<(rs.size())&&show;++i){
	//	printf("%s (%f) %f %f %f %f\n",names[int(rs[i][5])].c_str(),rs[i][4],rs[i][0],rs[i][1],rs[i][2],rs[i][3]);
	//}

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
void setting_python_path(const std::string& path){
	string pp = "PYTHONPATH="+path+":$PYTHONPATH";
	char p[1000] = {0};
	for(int i=0;i<pp.size();++i) p[i] = pp[i];
	putenv(p);
	Py_Initialize();
	PyRun_SimpleString("import caffe");
}
int main(int argc, char** argv){
	if(argc !=2){
		printf("./faster_rcnn_test 0/1\n");
		return 0;
	}

	setting_python_path("/Users/worm004/Projects/py-faster-rcnn/caffe-fast-rcnn/python:/Users/worm004/Projects/py-faster-rcnn/lib");

	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");
	map<string, TestParameter> maps;

	maps["faster-rcnn"] = TestParameter(
	       "../caffe_models/detection/VGG16_faster_rcnn_final.caffemodel",
	       "../caffe_models/detection/faster_rcnn_test.pt",
	       "../models/detection/faster-rcnn.net004.data",
	       "../models/detection/faster-rcnn.net004.net",
	       "../caffe_models/detection/voc.list",
	       122.7717, 115.9465, 102.9801, 1, 1, 1, 0.1, 20
	);
	bool show = atoi(argv[1]);
	string img_path = "../imgs/person.jpg";

	if(maps.find("faster-rcnn")!= maps.end()){
		printf("[TEST] [forwrad] %s\n","faster-rcnn");
		vector<vector<float> > caffe_rs, rs;
		caffe_forward(img_path, maps["faster-rcnn"], show, caffe_rs);
		//net004_forward(img_path,maps[argv[1]], show, rs);

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
