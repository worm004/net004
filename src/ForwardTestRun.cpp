#include <fstream>
#include "ForwardTestRun.h"
#include "DataLayer.h"
using namespace std;
ForwardTestRun::ForwardTestRun(){}
ForwardTestRun::ForwardTestRun(const JsonValue& j): ForwardBackwardRun(j){
}
void ForwardTestRun::operator()(Net004& net, int cur) {
	if(omit) return;
	if(cur%iter_interval != 0) return;
	printf("[%d]run: test\n",cur);

	DataLayer* img_layer = (DataLayer*)net[layer_map["img"]], *label_layer = (DataLayer*)net[layer_map["label"]];
	Layer* predict_layer = net[layer_map["predict"]];

	float * img_layer_data = img_layer->outputs[0].data, 
	      * label_layer_data = label_layer->outputs[0].data, 
	      * predict_layer_data = predict_layer->outputs[0].data;

	int batch_size = img_layer->n,
	    c = img_layer->c,
	    h = img_layer->h,
	    w = img_layer->w,
	    label_num = predict_layer->outputs[0].chw();

	int acc_top_1 = 0, all = 0;
	for(int i=0;i<iter;++i){
		input_data->fill_data(img_layer_data,batch_size,c,h,w,i*batch_size);
		input_data->fill_labels(label_layer_data,batch_size,i*batch_size);
		net.forward();
		for(int j=0;j<batch_size;++j){
			int max_label = -1;
			float max_val = 0.0;
			for(int k=0;k<label_num;++k){
				float val = predict_layer_data[j*label_num+k];
				//printf(" %.2f",val);
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
		//getchar();
		printf("[0->%d]top_1 %f\n",i*batch_size,float(acc_top_1)/float(all));
	}
}
void ForwardTestRun::init(const Net004& net){
	if(omit) return;
	//printf("init: ForwardTestRun\n");
	input_data->init();
}
