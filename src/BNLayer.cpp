#include "stdlib.h"
#include "BNLayer.h"

BNLayer::BNLayer(const std::string& name, float eps):Layer(name,"bn"),eps(eps){
}
BNLayer::~BNLayer(){
}
void BNLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());

	//show_inputs();
	// this should be done before forward
	// need refactoring
	float& s = scale.data[0];
	float *mean_data = mean.data, *var_data = variance.data;
	if(s != 0){
		int nchw =  mean.nchw();
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
	
	//show_outputs();
}
void BNLayer::backward(){
}
void BNLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: bn input blob number should be 1\n");
		exit(0);
	}

	// mean, variance and scale
	const Blob& ib = inputs[0];
	mean.set_shape(ib.c,1,1,1);
	variance.set_shape(ib.c,1,1,1);
	scale.set_shape(1,1,1,1);
	// output
	outputs.resize(1);
	outputs[0].set_shape(ib);

}
void BNLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: bn output blob number should be 1\n");
		exit(0);
	}
	// mean, variance and scale
	mean.alloc();
	variance.alloc();
	scale.alloc();
	// output
	outputs[0].alloc();
	//outputs[0].set_data(inputs[0].data);

}
void BNLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: bn input blob number should be 1\n");
		exit(0);
	}
	mean_dif.set_shape(mean);
	variance_dif.set_shape(variance);
	scale_dif.set_shape(scale);
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void BNLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: bn output blob number should be 1\n");
		exit(0);
	}
	mean_dif.alloc();
	variance_dif.alloc();
	scale_dif.alloc();
	output_difs[0].alloc();
}
void BNLayer::show() const{
	printf("[%s] name: %s\n", type.c_str(), name.c_str()); 
}
int BNLayer::parameter_number(){
	return mean.nchw() + variance.nchw() + scale.nchw();
}
