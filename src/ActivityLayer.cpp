#include "stdlib.h"
#include "ActivityLayer.h"
ActivityLayer::ActivityLayer(){}
ActivityLayer::ActivityLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	neg_slope  = attrs.jobj.at("neg_slope").jv.d;
	method  = attrs.jobj.at("method").jv.s;
	if(method != "relu"){
		printf("not implement %s method in activity layer\n",method.c_str());
		exit(0);
	}
	else {
		f = &ActivityLayer::forward_relu;
		bf = &ActivityLayer::backward_relu;
	}
}
void ActivityLayer::show(){
	Layer::show();
	printf("  (neg_slope) %g\n",neg_slope);
	printf("  (method) %s\n",method.c_str());
}
void ActivityLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	setup_outputs_data();
}
void ActivityLayer::forward(){
	//show_inputs();
	(this->*f)();
	//show_outputs();
}
void ActivityLayer::forward_relu(){
	float * odata = outputs[0].data, * idata = inputs[0].data;
	int nchw = inputs[0].nchw();
	for(int i=0;i<nchw;++i){
		if (idata[i] < 0.0f) odata[i] = neg_slope * idata[i];
		else odata[i] = idata[i];
	}
}
void ActivityLayer::backward_relu(){
	float *idata = inputs[0].data;
	float *input_data = diff_inputs[0].data, *output_data = diff_outputs[0].data;
	int nchw = diff_inputs[0].nchw();
	for(int i=0;i<nchw;++i){
		bool pos = idata[i] > 0;
		input_data[i] = output_data[i]*(pos + (!pos)*neg_slope);
	}
}
void ActivityLayer::backward(){
	//show_diff_outputs();
	(this->*bf)();
	//show_diff_inputs();
}
