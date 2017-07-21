#include "stdlib.h"
#include "ActivityLayer.h"

ActivityLayer::ActivityLayer(
	const std::string& name, 
	const std::string& method):
		method(method),
		Layer(name,"activity"){
}
ActivityLayer::~ActivityLayer(){
}

void ActivityLayer::forward_relu(){
	float * odata = outputs[0].data,
		* idata = inputs[0].data;
	int nchw = inputs[0].nchw();
	for(int i=0;i<nchw;++i) if (idata[i] < 0.0f) odata[i] = 0.0f;
}

void ActivityLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());

	if(method == "relu") forward_relu();
	else printf("not implemented %s in activit layer\n",method.c_str());

}
void ActivityLayer::backward(){
}
void ActivityLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: conv input blob number should be 1\n");
		exit(0);
	}
	const Blob& ib = inputs[0];

	// output
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(ib.n, ib.c, ib.h, ib.w);
	output_difs[0].set_shape(outputs[0]);
}
void ActivityLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: conv output blob number should be 1\n");
		exit(0);
	}
	outputs[0].set_data(inputs[0].data);

}
void ActivityLayer::show() const{
	printf("[%s] name: %s, method: %s\n", type.c_str(),name.c_str(), method.c_str());
}
