#include <fstream>
#include "ForwardTestRun.h"
#include "DataLayer.h"
using namespace std;
ForwardTestRun::ForwardTestRun(){}
ForwardTestRun::ForwardTestRun(const JsonValue& j): ForwardBackwardRun(j){
	cur_index = 0;
}
void ForwardTestRun::operator()(Net004& net, int cur) {
	if(omit) return;
	if(cur%iter_interval != 0) return;
	//printf("[%d]run: test\n",cur);

	DataLayer* img_layer = (DataLayer*)net[layer_map["img"]], *label_layer = (DataLayer*)net[layer_map["label"]];
	Layer* predict_layer = net[layer_map["predict"]], * loss_layer = net[layer_map["loss"]];

	float * img_layer_data = img_layer->outputs[0].data, 
	      * label_layer_data = label_layer->outputs[0].data, 
	      * predict_layer_data = predict_layer->outputs[0].data,
	      * loss_layer_data = loss_layer->outputs[0].data;

	int batch_size = img_layer->n,
	    c = img_layer->c,
	    h = img_layer->h,
	    w = img_layer->w,
	    label_num = predict_layer->outputs[0].chw();

	int acc_top_1 = 0, all = 0;
	float loss = 0.0f;
	for(int i=0;i<iter;++i){
		input_data->fill_data(img_layer_data,batch_size,c,h,w,cur_index);
		input_data->fill_labels(label_layer_data,batch_size,cur_index);
		cur_index += batch_size;
		net.forward();
		loss += loss_layer_data[0];
		for(int j=0;j<batch_size;++j){
			int max_label = -1;
			float max_val = 0.0;
			for(int k=0;k<label_num;++k){
				float val = predict_layer_data[j*label_num+k];
				if( val > max_val){
					max_val = val;
					max_label = k;
				}
			}
			if (max_label == label_layer_data[j]) ++acc_top_1;
			++all;
			//printf("\n[%d] predict: %d (%s), ground truth: %d (%s)\n",
			//	i*batch_size+j,
			//	max_label,input_data->label_name(max_label).c_str(),
			//	int(label_layer_data[j]),input_data->label_name(label_layer_data[j]).c_str());
		}
	}
	printf("[iter %07d] [test] [data index %08d - %08d] [test_loss %.3f]\n",cur,cur_index - iter*batch_size, cur_index-1,loss/iter);
	printf("[iter %07d] [test] [data index %08d - %08d] [accuracy(top1) %.3f]\n",cur,cur_index - iter*batch_size, cur_index-1,float(acc_top_1)/float(all));
	//printf("Accuracy(top1) in [%d,%d): %f\n",cur_index-iter*batch_size,cur_index,float(acc_top_1)/float(all));
	//fflush(stdout);
}
void ForwardTestRun::init(const Net004& net){
	if(omit) return;
	input_data->init();
}
