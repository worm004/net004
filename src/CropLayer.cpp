#include "stdlib.h"
#include "CropLayer.h"
CropLayer::CropLayer(const std::string&name, int axis, const std::vector<int>& offset, const std::vector<std::string>& names):Layer(name,"crop"), axis(axis){
	for(int i=0;i<offset.size();++i)
		this->offset.push_back(offset[i]);

	for(int i=0;i<names.size();++i)
		order[names[i]] = i;
	if(is_train) input_difs.resize(names.size());
	inputs.resize(names.size());
}
CropLayer::~CropLayer(){
}
void CropLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());
	//show_inputs();

	//show_outputs();
}
void CropLayer::backward(){
	printf("backward: %s %s\n",type.c_str(), name.c_str());
}
void CropLayer::setup_shape(){
	if(inputs.size()!=2){
		printf("error: crop input blob number should be 2\n");
		exit(0);
	}
	// output
	outputs.resize(1);
	outputs[0].set_shape(inputs[1]);
}
void CropLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: crop output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();

}
void CropLayer::setup_dif_shape(){
	if(input_difs.size()!=2){
		printf("error: crop input blob number should be 2\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void CropLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: crop output blob number should be 1\n");
		exit(0);
	}
	output_difs[0].alloc();
}

void CropLayer::show()const {
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

