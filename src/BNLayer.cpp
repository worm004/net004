#include <cmath>
#include "stdlib.h"
#include "BNLayer.h"
BNLayer::BNLayer(){}
BNLayer::BNLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	eps = attrs.jobj.at("eps").jv.d;
}
void BNLayer::show(){
	Layer::show();
	printf("  (eps) %g\n", eps);
}
void BNLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	setup_outputs_data();
}
void BNLayer::forward(){
	float& s = params["scale"].data[0];
	float *mean_data = params["mean"].data, *var_data = params["variance"].data;
	if(s != 0){
		int nchw =  params["mean"].nchw();
		for(int i=0;i<nchw;++i){
			mean_data[i] /= s;
			var_data[i] /= s;
		}
		s = 0;
	}
	int n = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();
	float *input_data = inputs[0].data, *output_data = outputs[0].data;
	for(int i=0;i<n;++i){
		for(int j=0;j<c;++j){
			float m = mean_data[j];
			float v = var_data[j];
			float rsqrt_v = 1/sqrt(eps + v);
			for(int k=0;k<hw;++k){
				int index = i*c*hw + j*hw + k;
				output_data[index] = (input_data[index] - m)*rsqrt_v;
			}
		}
	}
}
