#include "stdlib.h"
#include "ProposalLayer.h"

ProposalLayer::ProposalLayer(
	const std::string&name, 
	int feat_stride, 
	const std::vector<std::string>& names,
	const std::string& method):
		feat_stride(feat_stride),
		method(method),
		Layer(name,"proposal"){

	for(int i=0;i<names.size();++i)
		order[names[i]] = i;
	if(is_train) input_difs.resize(names.size());
	inputs.resize(names.size());
}
ProposalLayer::~ProposalLayer(){
}
void ProposalLayer::forward(){
	//printf("forward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
	//show_inputs();

	//show_outputs();
}

void ProposalLayer::backward(){
	printf("backward: %s %s %s\n",type.c_str(), name.c_str(), method.c_str());
}
void ProposalLayer::show()const {
	printf("[%s%s] name: %s, feat_stride: %d\n",
			type.c_str(),("+"+method).c_str(), 
			name.c_str(),
			feat_stride);
}
void ProposalLayer::setup_shape(){
	if(inputs.size()<=1){
		printf("error: proposal input blob number should be > 1\n");
		exit(0);
	}
	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	outputs[0].set_shape(ib.n,5,ib.h,ib.w);
}
void ProposalLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: proposal output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
}
void ProposalLayer::setup_dif_shape(){
	if(input_difs.size() <=1){
		printf("error: proposal input blob number should be > 1\n");
		exit(0);
	}
	output_difs.resize(1);
}
void ProposalLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: proposal output blob number should be 1\n");
		exit(0);
	}
}
