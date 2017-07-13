#include "stdlib.h"
#include "PoolLayer.h"

PoolLayer::PoolLayer(
	const std::string&name, 
	int kernel, 
	int stride, 
	int padding, 
	const std::string& method):
		kernel(kernel),
		stride(stride),
		padding(padding),
		method(method),
		Layer(name,"pool"){
}
PoolLayer::~PoolLayer(){
}
void PoolLayer::forward(){
}
void PoolLayer::backward(){
}
void PoolLayer::show()const {
	printf("[%s%s] name: %s, kernel: %d, stride: %d, padding: %d\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str(),
			kernel,stride,padding);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
}
void PoolLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: pool input blob number should be 1\n");
		exit(0);
	}

	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	output_difs.resize(1);
	int oh = Layer::i2o_ceil(ib.h,kernel,stride,padding),
	    ow = Layer::i2o_ceil(ib.w,kernel,stride,padding);
	outputs[0].set_shape(ib.n, ib.c, oh, ow);
	output_difs[0].set_shape(outputs[0]);
}
void PoolLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: pool output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	output_difs[0].alloc();
}
