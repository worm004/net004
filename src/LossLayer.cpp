#include "stdlib.h"
#include "LossLayer.h"
LossLayer::LossLayer(const std::string&name, 
			const std::string& method):
				method(method),
				Layer(name,"loss"){
}
LossLayer::~LossLayer(){
}
void LossLayer::forward(){
}
void LossLayer::backward(){
}
void LossLayer::show()const {
	printf("[%s%s] name: %s\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str());
}
void LossLayer::setup_shape(){
	if( (inputs.size()!=2) || (input_difs.size()!=2)){
		printf("error: loss input blob number should be 2\n");
		exit(0);
	}
	loss = 0.0f;
}
void LossLayer::setup_data(){
}
