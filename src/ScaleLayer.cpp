#include "stdlib.h"
#include "ScaleLayer.h"
ScaleLayer::ScaleLayer(const std::string& name):
	Layer(name,"scale"){
}
ScaleLayer::~ScaleLayer(){
}
void ScaleLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());

	float *weight_data = weight.data, *bias_data = bias.data;

	int n = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();

	float *input_data = inputs[0].data, *output_data = outputs[0].data;

	//show_inputs();
	if(bias_data){
		for(int i=0;i<n;++i){
			for(int j=0;j<c;++j){
				float s = weight_data[j];
				float b = bias_data[j];
				for(int k=0;k<hw;++k){
					int index = i*c*hw + j*hw + k;
					output_data[index] = input_data[index] * s + b;
				}
			}
		}
	}
	else{
		for(int i=0;i<n;++i){
			for(int j=0;j<c;++j){
				float s = weight_data[j];
				for(int k=0;k<hw;++k){
					int index = i*c*hw + j*hw + k;
					output_data[index] = input_data[index] * s;
				}
			}
		}
	}
	//show_outputs();

}
void ScaleLayer::backward(){
}
void ScaleLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: scale input blob number should be 1\n");
		exit(0);
	}

	// weight and bias
	const Blob& ib = inputs[0];
	weight.set_shape(ib.c,1,1,1);
	weight_dif.set_shape(weight);

	// output
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(ib);
	output_difs[0].set_shape(outputs[0]);
}
void ScaleLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: scale output blob number should be 1\n");
		exit(0);
	}
	// mean and variance
	weight.alloc();
	weight_dif.alloc();

	// output
	outputs[0].set_data(inputs[0].data);
	output_difs[0].alloc();
}
void ScaleLayer::show() const{
	printf("[%s%s] name: %s\n", type.c_str(),bias.data?"+bias":"", name.c_str()); 
}
int ScaleLayer::parameter_number(){
	return weight.nchw() + bias.nchw();
}
