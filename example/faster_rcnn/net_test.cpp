#include "caffe/caffe.hpp"
#include "faster_rcnn_tool.h"
#include "Net004.h"
#include "Parser.h"
#include "DataLayer.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace std;
using namespace cv;
using namespace caffe;

void caffe_forward(const Mat&img, const FasterRCNNConfig& config, bool show, vector<vector<float>>& dets, float scale);
void net004_forward(const Mat&img, const FasterRCNNConfig& config, bool show, vector<vector<float>>& dets, float scale);

int main(int argc, char** argv){
	if(argc !=2){
		printf("./faster_rcnn_test 0/1\n");
		return 0;
	}
	google::InitGoogleLogging(argv[0]);
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	setting_python_path("/Users/worm004/Projects/py-faster-rcnn/caffe-fast-rcnn/python:/Users/worm004/Projects/py-faster-rcnn/lib");
	FasterRCNNConfig config;
	set_config(config);

	string img_path = "../imgs/person.jpg";
	Mat img = imread(img_path), rimg;
	float scale;
	preprocess_img(img, rimg, scale,
			config.target_size, config.max_size,
			config.mean[0], config.mean[1], config.mean[2]);

	bool show = atoi(argv[1]);
	printf("[TEST] [forwrad] %s\n","faster-rcnn");
	vector<vector<float> > caffe_rs, rs;
	caffe_forward(rimg, config, show, caffe_rs, scale);
	net004_forward(rimg, config, show, rs, scale);

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

	return 0;
}
void net004_forward(const Mat&img, const FasterRCNNConfig& config, bool show, vector<vector<float>>& dets, float scale){
	if(show) printf("net004 forwarding ...\n");
	Net004 net;
	Parser parser;
	parser.set_input_size("data",1,img.channels(),img.rows,img.cols);
	auto t1 = now();
	parser.read(config.net004_net_path, config.net004_model_path, &net);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	//net.show();

	int c = img.channels(), h = img.rows, w = img.cols;
	Layers & ls = net.ls;
	float *data0 = (float*)(DataLayer*)ls["data"]->outputs[0].data,
	      *data1 = (float*)(DataLayer*)ls["im_info"]->outputs[0].data,
	      *data = (float*)img.data;

	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j)
	for(int k=0;k<3;++k)
		data0[(i*w+j) + h*w*k] = data[(i*w+j)*3+k]; 

	data1[0] = img.rows;
	data1[1] = img.cols;
	data1[2] = scale;

	t1 = now();
	net.forward();
	t2 = now();
	if(show){
		cout<<"forward: "<<cal_duration(t1,t2)<<endl;
		//net.show();
	}

	vector<vector<float> > pred_boxes;

	float *rois_data = net.ls["proposal"]->outputs[0].data;
	float *bbox_pred_data = net.ls["bbox_pred"]->outputs[0].data;
	float *score_data = net.ls["cls_prob"]->outputs[0].data;
	bbox_transform_inv(rois_data, bbox_pred_data, net.ls["proposal"]->outputs[0].c/5, config.cnum, pred_boxes, scale);
	bbox_bound(pred_boxes,float(img.cols/scale-1),float(img.rows/scale-1));
	nms_all_class(pred_boxes,score_data,dets,config.cnum,config.nms_thres,config.conf_thres);
	
	if(show){
		vector<string> names(config.cnum);
		ifstream file(config.list_path);
		for(int i=0;i<config.cnum;++i)
			file >> names[i];
		for(int i=0;i<dets.size();++i){
			printf("%s (%f) %f %f %f %f\n",names[int(dets[i][5])].c_str(),dets[i][4],dets[i][0],dets[i][1],dets[i][2],dets[i][3]);
		}
	}

}
void caffe_forward(const Mat&img, const FasterRCNNConfig& config, bool show, vector<vector<float>>& dets,float scale){
	if(show) printf("caffe forwarding ...\n");

  	std::shared_ptr<caffe::Net<float> > net;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	auto t1 = now();
	net = make_shared<caffe::Net<float>> (config.caffe_net_path, caffe::TEST);
	net->CopyTrainedLayersFrom(config.caffe_model_path);
	auto t2 = now();
	if(show) cout<<"read: "<<cal_duration(t1,t2)<<endl;

	int c = img.channels(), h = img.rows, w = img.cols;
	net->input_blobs()[0]->Reshape(1, c, h, w);
	net->Reshape();

	float * input_blob0_data = net->input_blobs()[0]->mutable_cpu_data(), 
		* input_blob1_data = net->input_blobs()[1]->mutable_cpu_data(),
		* data = (float*)img.data;

	for(int i=0;i<h;++i)
	for(int j=0;j<w;++j)
	for(int k=0;k<3;++k)
		input_blob0_data[(i*w+j) + h*w*k] = data[(i*w+j)*3+k]; 

	input_blob1_data[0] = img.rows;
	input_blob1_data[1] = img.cols;
	input_blob1_data[2] = scale;

	float loss;
	t1 = now();
	net->ForwardPrefilled(&loss)[0];
	t2 = now();
	if(show) cout<<"forward: "<<cal_duration(t1,t2)<<endl;

	const caffe::shared_ptr<caffe::Blob<float> > roi_blob = net->blob_by_name("rois");
	const caffe::shared_ptr<caffe::Blob<float> > bbox_pred_blob = net->blob_by_name("bbox_pred");
	const caffe::shared_ptr<caffe::Blob<float> > score_blob = net->blob_by_name("cls_prob");

	vector<vector<float> > pred_boxes;
	bbox_transform_inv(roi_blob->cpu_data(), bbox_pred_blob->cpu_data(), roi_blob->num(), config.cnum, pred_boxes, scale);
	bbox_bound(pred_boxes,float(img.cols/scale-1),float(img.rows/scale-1));
	nms_all_class(pred_boxes,score_blob->cpu_data(),dets,config.cnum,config.nms_thres,config.conf_thres);
	
	if(show){
		vector<string> names(config.cnum);
		ifstream file(config.list_path);
		for(int i=0;i<config.cnum;++i)
			file >> names[i];
		for(int i=0;i<dets.size();++i){
			printf("%s (%f) %f %f %f %f\n",names[int(dets[i][5])].c_str(),dets[i][4],dets[i][0],dets[i][1],dets[i][2],dets[i][3]);
		}
	}

}
