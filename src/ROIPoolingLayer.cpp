#include "stdlib.h"
#include "ROIPoolingLayer.h"
ROIPoolingLayer::ROIPoolingLayer(
	const std::string&name, int h, int w, float scale, const std::vector<std::string>& names):
		h(h),
		w(w),
		scale(scale),
		Layer(name,"roipooling"){

	for(int i=0;i<names.size();++i)
		order[names[i]] = i;
	if(is_train) input_difs.resize(names.size());
	inputs.resize(names.size());
}
ROIPoolingLayer::~ROIPoolingLayer(){
}
void ROIPoolingLayer::forward(){
	//printf("forward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
	//show_inputs();

	//show_outputs();
}

void ROIPoolingLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void ROIPoolingLayer::show()const {
	printf("[%s] name: %s, h: %d, w: %d, scale: %f\n",
			type.c_str(), 
			name.c_str(),
			h,w,scale);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
}
void ROIPoolingLayer::setup_shape(){
	if(inputs.size()!=2){
		printf("error: roipool input blob number should be 2\n");
		exit(0);
	}
	// output
	const Blob& ib0 = inputs[0];
	const Blob& ib1 = inputs[1];
	outputs.resize(1);
	outputs[0].set_shape(ib1.n, ib0.c, h, w);
}
void ROIPoolingLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: roipool output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
}
void ROIPoolingLayer::setup_dif_shape(){
	if(input_difs.size()!=2){
		printf("error: roipool input blob number should be 2\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void ROIPoolingLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: pool output blob number should be 1\n");
		exit(0);
	}
	output_difs[0].alloc();
}
