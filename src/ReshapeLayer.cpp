#include "stdlib.h"
#include "ReshapeLayer.h"
ReshapeLayer::ReshapeLayer(const std::string& name, const std::vector<int>& p4):
		Layer(name,"reshape") {
	for(int i=0;i<4;++i)
		shape.push_back(p4[i]);
}
ReshapeLayer::~ReshapeLayer(){
}
void ReshapeLayer::forward(){
}
void ReshapeLayer::backward(){
}
void ReshapeLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: reshape input blob number should be 1\n");
		exit(0);
	}
	const Blob& ib = inputs[0];
	outputs.resize(1);

	if(shape[0] == 0) shape[0] = inputs[0].n;
	if(shape[1] == 0) shape[1] = inputs[0].c;
	if(shape[2] == 0) shape[2] = inputs[0].h;
	if(shape[3] == 0) shape[3] = inputs[0].w;

	while(1){
		int num = 0;
		for(int i=0;i<4;++i){
			if(shape[i] == -1)
				++num;
		}
		if(num == 1){
			for(int i=0;i<4;++i){
				if(shape[i] == -1){
					int total = 1;
					for(int j=0;j<4;++j){
						if(j == i) continue;
						total *= shape[j];
					}
					shape[i] = inputs[0].nchw()/total;
				}
			}
		}
		else if(num == 0)break;
		else{
			printf("err: cannot be two or more -1 in reshape\n");
			exit(0);
		}
	}

	outputs[0].set_shape(shape[0],shape[1],shape[2],shape[3]);
}
void ReshapeLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: reshape output blob number should be 1\n");
		exit(0);
	}
	outputs[0].set_data(inputs[0].data);
}
void ReshapeLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: reshape input blob number should be 1\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void ReshapeLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: reshape output blob number should be 1\n");
		exit(0);
	}
}
void ReshapeLayer::show() const{
	printf("[%s] name: %s\n", type.c_str(),name.c_str());
}
