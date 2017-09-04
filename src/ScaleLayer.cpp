#include <cmath>
#include "stdlib.h"
#include "ScaleLayer.h"
ScaleLayer::ScaleLayer(){}
ScaleLayer::ScaleLayer(const LayerUnit& u):Layer(u){
	float v;
	u.geta("bias",v); bias = v;
	if(bias) f = &ScaleLayer::forward_bias;
	else f = &ScaleLayer::forward_nbias;
}
void ScaleLayer::show(){
	Layer::show();
	printf("  (bias) %d\n",int(bias));
}
void ScaleLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	setup_outputs_data();
}
void ScaleLayer::forward(){
	(this->*f)();
}
void ScaleLayer::forward_bias(){
	int n = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();
	float *input_data = inputs[0].data, *output_data = outputs[0].data;
	float *weight_data = params["scale"].data, *bias_data = params["bias"].data;
	for(int i=0;i<n;++i)
		for(int j=0;j<c;++j){
			float s = weight_data[j], b = bias_data[j];
			for(int k=0;k<hw;++k){
				int index = i*c*hw + j*hw + k;
				output_data[index] = input_data[index] * s + b;
			}
		}
}
void ScaleLayer::forward_nbias(){
	int n = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();
	float *input_data = inputs[0].data, *output_data = outputs[0].data;
	float *weight_data = params["scale"].data;
	for(int i=0;i<n;++i)
		for(int j=0;j<c;++j){
			float s = weight_data[j];
			for(int k=0;k<hw;++k){
				int index = i*c*hw + j*hw + k;
				output_data[index] = input_data[index] * s;
			}
		}
}
