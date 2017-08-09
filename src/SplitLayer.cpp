#include "stdlib.h"
#include "SplitLayer.h"
SplitLayer::SplitLayer( const std::string&name):
		Layer(name,"split"){
}
SplitLayer::~SplitLayer(){
}
void SplitLayer::forward(){
	//printf("forward: %s %s\n",type.c_str(), name.c_str());
	//show_inputs();
}
void SplitLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void SplitLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: split input blob number should be 1\n");
		exit(0);
	}
	// output
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(inputs[0]);
	output_difs[0].set_shape(inputs[0]);
}
void SplitLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: split output blob number should be 1\n");
		exit(0);
	}

	// output
	outputs[0].set_data(inputs[0].data);
	output_difs[0].set_data(input_difs[0].data);
}
void SplitLayer::show() const{
	printf("[%s] name: %s\n", type.c_str(),name.c_str() );
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}

	if(outputs.size() != 0){
		printf("\toutput: ");
		outputs[0].show();
	}
}
int SplitLayer::parameter_number(){
	return 0;
}
