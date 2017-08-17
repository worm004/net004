#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Python.h"
struct FasterRCNNConfig{
	std::string list_path;
	int cnum;

	float mean[3];//rgb
	std::string caffe_model_path, caffe_net_path;
	float nms_thres, conf_thres;
	int target_size, max_size;

	std::string net004_model_path, net004_net_path;
};

void preprocess_img(const cv::Mat& src, cv::Mat& des, float &scale, int target_scale, int max_size, float r, float g, float b);
void setting_python_path(const std::string& path);
void bbox_transform_inv(const float *boxes, const float * deltas, int n, int cnum, std::vector<std::vector<float> >& pred_boxes, float scale);
void bbox_bound(std::vector<std::vector<float> >& pred_boxes, float w, float h);
void nms(std::vector<std::vector<float> >& rs, float T);
void nms_all_class(const std::vector<std::vector<float> >& boxes, const float* score_data, 
			std::vector<std::vector<float> >& dets,
			int cnum, float nms_thres, float conf_thres);
