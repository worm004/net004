#include "stdlib.h"
#include "EltwiseLayer.h"
EltwiseLayer::EltwiseLayer(const std::string& name, const std::string& method):method(method),Layer(name,"eltwise"){
}
EltwiseLayer::~EltwiseLayer(){
}
void EltwiseLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	//show_inputs();
	if(method == "sum"){
		int nchw = outputs[0].nchw();
		float *b0 = inputs[0].data, *b1 = inputs[1].data, *output_data = outputs[0].data;
		for(int i=0;i<nchw;++i)
			output_data[i] = b0[i] + b1[i];
	}
	//show_outputs();
}
void EltwiseLayer::backward(){
}
void EltwiseLayer::setup_shape(){
	if(inputs.size()!=2){
		printf("error: elewise input blob number should be 2\n");
		exit(0);
	}
	else if(!inputs[0].is_shape_same(inputs[1])){
		printf("error: eltwise input 2 blobs should have same shape\n");
		exit(0);
	}

	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	outputs[0].set_shape(ib);

	if(is_train){
		if(input_difs.size()!=2){
			printf("error: elewise input blob number should be 2\n");
			exit(0);
		}
		else if(!input_difs[0].is_shape_same(input_difs[1])){
			printf("error: eltwise input 2 blobs should have same shape\n");
			exit(0);
		}
		output_difs.resize(1);
		output_difs[0].set_shape(outputs[0]);
	}
}
void EltwiseLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: eltwise output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();

	if(is_train){
		if(output_difs.size()!=1){
			printf("error: eltwise output blob number should be 1\n");
			exit(0);
		}
		output_difs[0].alloc();
	}
}
void EltwiseLayer::show() const{
	printf("[%s+%s] name: %s\n", type.c_str(),method.c_str(), name.c_str()); 
}
int EltwiseLayer::parameter_number(){
	return 0;
}
