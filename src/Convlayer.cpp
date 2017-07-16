#include "stdlib.h"
#include "ConvLayer.h"

ConvLayer::ConvLayer(
	const std::string&name, 
	int filters, 
	int kernel, 
	int stride, 
	int padding, 
	const std::string& activity):
		kernel(kernel), 
		filters(filters), 
		padding(padding), 
		stride(stride), 
		activity(activity), 
		Layer(name,"conv"){
}

ConvLayer::~ConvLayer(){
}

void ConvLayer::forward(){
}

void ConvLayer::backward(){
}

void ConvLayer::show()const {
	printf("[%s%s] name: %s, filters: %d, kernel: %d, stride: %d, padding: %d\n",
		type.c_str(),activity.empty()?"":("+"+activity).c_str(), 
		name.c_str(),
		filters,kernel,stride,padding);

	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	//printf("\tinput dif:");
	//input_difs[0].show();

	if(bias.total() != 0){
		printf("\tbias: ");
		bias.show();
	}
	if(weight.total() != 0){
		//printf("\tbias dif: ");
		//bias_dif.show();
		printf("\tweight: ");
		weight.show();
	}
	//printf("\tweight dif: ");
	//weight_dif.show();

	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}
	//printf("\toutput dif:");
	//output_difs[0].show();
}

int ConvLayer::parameter_number(){
	return weight.total() + bias.total();
}

void ConvLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	weight.alloc();
	weight_dif.alloc();
	bias.alloc();
	bias_dif.alloc();

	// output
	outputs[0].alloc();
	output_difs[0].alloc();
}

void ConvLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: conv input blob number should be 1\n");
		exit(0);
	}
	// weight and bias
	const Blob& ib = inputs[0];
	weight.set_shape(1,ib.c * filters, kernel, kernel);
	weight_dif.set_shape(weight);
	bias.set_shape(1,filters,1,1);
	bias_dif.set_shape(bias);

	// output
	outputs.resize(1);
	output_difs.resize(1);
	int oh = Layer::i2o_floor(ib.h,kernel,stride,padding),
	    ow = Layer::i2o_floor(ib.w,kernel,stride,padding);
	outputs[0].set_shape(ib.n, filters, oh, ow);
	output_difs[0].set_shape(outputs[0]);
}
