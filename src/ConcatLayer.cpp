#include "stdlib.h"
#include "ConcatLayer.h"
ConcatLayer::ConcatLayer(const std::string&name):Layer(name,"concat"){
}
ConcatLayer::~ConcatLayer(){
}
void ConcatLayer::forward(){
}
void ConcatLayer::backward(){
}
void ConcatLayer::setup_shape(){
	if( (inputs.size()<=1) || (input_difs.size()<=1)){
		printf("error: concat input blob number should be > 1\n");
		exit(0);
	}
	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	output_difs.resize(1);

	int c = 0, h = inputs[0].h, w = inputs[0].w;
	for (const auto& i: inputs){
		c += i.c;
		if((i.h != h) || (i.w != w)){
			printf("error: concat inputs should have same w and h\n");
			exit(0);
		}
	}
	outputs[0].set_shape(ib.n, c, ib.h, ib.w);
	output_difs[0].set_shape(outputs[0]);
}
void ConcatLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: concat output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	output_difs[0].alloc();
}
void ConcatLayer::show()const {
	printf("[%s] name: %s\n", type.c_str(), name.c_str());

	if(inputs.size()){
		printf("\tinput: \n");
		for(int i=0;i<inputs.size();++i){
			printf("\t");
			inputs[i].show();
		}
	}

	if(outputs.size()){
		printf("\toutput: \n");
		for(int i=0;i<outputs.size();++i){
			printf("\t");
			outputs[i].show();
		}
	}
}