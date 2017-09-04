#include "faster_rcnn_tool.h"
using namespace cv;
using namespace std;
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
	int max_s = max(src.cols,src.rows);
	scale = float(target_scale)/float(min(src.cols,src.rows));
	if(int(scale * max_s+0.5) > max_size) scale = float(max_size) / float(max_s);
	resize(des,des,Size(),scale,scale,CV_INTER_LINEAR);
}
void setting_python_path(const std::string& path){
	std::string pp = "PYTHONPATH="+path+":$PYTHONPATH";
	char p[1000] = {0};
	for(int i=0;i<pp.size();++i) p[i] = pp[i];
	putenv(p);
	Py_Initialize();
	PyRun_SimpleString("import caffe");
}


void bbox_transform_inv(const float *boxes, const float * deltas, int n, int cnum, std::vector<std::vector<float> >& pred_boxes, float scale){
	int c = (cnum+1)*4;
	pred_boxes.resize(n,vector<float>(c));

	for(int i=0;i<n;++i){
		float width = (boxes[i*5+1+2] - boxes[i*5+1+0])/scale + 1.0f,
			height = (boxes[i*5+1+3] - boxes[i*5+1+1])/scale + 1.0f,
			cx = boxes[i*5+1+0]/scale + 0.5*width,
			cy = boxes[i*5+1+1]/scale + 0.5*height;

		for(int j=0;j<c;j+=4){
			float dx = deltas[i*c+j],
				dy = deltas[i*c+j+1],
				dw = deltas[i*c+j+2],
				dh = deltas[i*c+j+3];
			//printf("%f %f %f %f",dx,dy,dw,dh);

			float pred_cx = dx * width + cx,
			      pred_cy = dy * height + cy,
			      pred_w = exp(dw) * width,
			      pred_h = exp(dh) * height;
			//printf("%f %f %f %f",pred_cx, pred_cy, pred_w, pred_h);

			pred_boxes[i][j] =  pred_cx - 0.5*pred_w;
			pred_boxes[i][j+1] = pred_cy - 0.5*pred_h;
			pred_boxes[i][j+2] = pred_cx + 0.5*pred_w;
			pred_boxes[i][j+3] = pred_cy + 0.5*pred_h;
		}
	}
}
void bbox_bound(std::vector<std::vector<float> >& pred_boxes, float w, float h){
	for(int i=0;i<pred_boxes.size();++i){
		for(int j=0;j<pred_boxes[i].size();j+=4){
			pred_boxes[i][j] =   std::max(pred_boxes[i][j],0.0f);
			pred_boxes[i][j+1] = std::max(pred_boxes[i][j+1],0.0f);
			pred_boxes[i][j+2] = std::min(pred_boxes[i][j+2],w);
			pred_boxes[i][j+3] = std::min(pred_boxes[i][j+3],h);
		}
	}
}
void nms(vector<vector<float> >& rs, float T){
	int count = rs.size();
	if(count == 0) return;
	vector<int> indics(count);
	for(int i=0;i<count;++i) indics[i] = i;
	sort(indics.begin(), indics.end(),
		[&](int i, int j) {return rs[i][4] > rs[j][4];});
	vector<vector<float> > temp;
	vector<bool> deletes(count,false);
	for(int i=0;i<count;++i){
		int index = indics[i];
		temp.push_back({rs[index][0], rs[index][1], rs[index][2], rs[index][3], 
				(rs[index][2]-rs[index][0]+1) * (rs[index][3] - rs[index][1] + 1), 
				rs[index][4]});
	}
	rs.clear();
	for(int i=0;i<count;++i){
		if(deletes[i]) continue;
		rs.push_back({temp[i][0], temp[i][1], temp[i][2], temp[i][3],temp[i][5]});
		for(int j=i+1;j<count;++j){
			if(deletes[j]) continue;
			float l = std::max(temp[i][0], temp[j][0]),
			      t = std::max(temp[i][1], temp[j][1]),
			      r = std::min(temp[i][2], temp[j][2]),
			      b = std::min(temp[i][3], temp[j][3]),
			      area = (r-l) * (b-t);
			if ((r-l <=0) || (b-t <=0) ||area / (temp[i][4] + temp[j][4] - area) <= T){
				continue;
			}
			deletes[j] = true;
		}
	}
}
void nms_all_class(const std::vector<std::vector<float> >& boxes, const float* score_data, 
			std::vector<std::vector<float> >& dets,
			int cnum, float nms_thres, float conf_thres){
	dets.clear();
	for(int i=0;i<cnum;++i){
		int n = boxes.size();
		vector<vector<float> > temp;
		for(int j=0;j<n;++j){
			if(score_data[j*(cnum+1)+i+1] < conf_thres) continue;
			temp.push_back({
				boxes[j][(i+1)*4],
				boxes[j][(i+1)*4+1],
				boxes[j][(i+1)*4+2],
				boxes[j][(i+1)*4+3],
				score_data[j*(cnum+1)+i+1]
				});
		}
		nms(temp, nms_thres);
		for(int j=0;j<temp.size();++j)
			dets.push_back({temp[j][0],temp[j][1],temp[j][2],temp[j][3],temp[j][4],float(i)});
	}

}
void set_config(FasterRCNNConfig & config){
	config.list_path = "../caffe_models/detection/voc.list";
	config.cnum = 20;
	config.mean[0] = 122.7717f;
	config.mean[1] = 115.9465f;
	config.mean[2] = 102.9801f;
	config.caffe_model_path = "../caffe_models/detection/VGG16_faster_rcnn_final.caffemodel";
	config.caffe_net_path = "../caffe_models/detection/faster_rcnn_test.pt";
	config.net004_model_path = "../models2/faster_rcnn.net004.data";
	config.net004_net_path = "../models2/faster_rcnn.net004.net";
	config.nms_thres = 0.3f;
	config.conf_thres = 0.8f;
	config.target_size = 600;
	config.max_size = 700;
}
