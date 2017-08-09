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

	if(is_train){
		// do this when backward
		//for(int i=0;i<nchw;++i) 
		//	mask[i] = 1;

		for(int i=0;i<nchw;++i)
			if (idata[i] < 0.0f){
				odata[i] = 0.0f;
				mask[i] = 0;
			}
	}
	else{
		for(int i=0;i<nchw;++i)
			if (idata[i] < 0.0f)
				odata[i] = 0.0f;
	}
}

void ActivityLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());

	//show_inputs();
	if(method == "relu") forward_relu();
	else printf("not implemented %s in activit layer\n",method.c_str());
	//show_outputs();

}
void ActivityLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void ActivityLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: acvtivity input blob number should be 1\n");
		exit(0);
	}
	const Blob& ib = inputs[0];
	outputs.resize(1);
	outputs[0].set_shape(ib);
}
void ActivityLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: activity output blob number should be 1\n");
		exit(0);
	}

	outputs[0].set_data(inputs[0].data);

}
void ActivityLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: acvtivity input blob number should be 1\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void ActivityLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: activity output blob number should be 1\n");
		exit(0);
	}
	mask = new bool[outputs[0].nchw()];
	memset(mask,0,sizeof(bool)*outputs[0].nchw());
}
void ActivityLayer::show() const{
	printf("[%s] name: %s, method: %s\n", type.c_str(),name.c_str(), method.c_str());
}
