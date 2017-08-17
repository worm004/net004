#include "caffe/caffe.hpp"
#include "faster_rcnn_tool.h"
#define now() (std::chrono::high_resolution_clock::now())
#define cal_duration(t1,t2) (std::chrono::duration_cast<std::chrono::milliseconds>((t2) - (t1)).count())
using namespace std;
using namespace cv;
using namespace caffe;

void set_config(FasterRCNNConfig & config);
void caffe_forward(const Mat&img, const FasterRCNNConfig& config, bool show, vector<vector<float>>& det, float scale);

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
	vector<vector<float> > caffe_rs;
	caffe_forward(rimg, config, show, caffe_rs,scale);

	return 0;
}
void set_config(FasterRCNNConfig & config){
	config.list_path = "../caffe_models/detection/voc.list";
	config.cnum = 20;
	config.mean[0] = 122.7717f;
	config.mean[1] = 115.9465f;
	config.mean[2] = 102.9801f;
	config.caffe_model_path = "../caffe_models/detection/VGG16_faster_rcnn_final.caffemodel";
	config.caffe_net_path = "../caffe_models/detection/faster_rcnn_test.pt";
	config.net004_model_path = "../models/detection/faster-rcnn.net004.data";
	config.net004_net_path = "../models/detection/faster-rcnn.net004.net";
	config.nms_thres = 0.3f;
	config.conf_thres = 0.8f;
	config.target_size = 600;
	config.max_size = 1000;
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
