#include "stdlib.h"
#include "LRNLayer.h"
LRNLayer::LRNLayer(
	const std::string&name, 
	int n, 
	float beta, 
	float alpha):
		n(n),
		beta(beta),
		alpha(alpha),
		Layer(name,"lrn"){
}
LRNLayer::~LRNLayer(){
}
void LRNLayer::forward(){
}
void LRNLayer::backward(){
}
void LRNLayer::show()const {
	printf("[%s] name: %s, n: %d, alpha: %.4f, beta: %.4f\n",
			type.c_str(), 
			name.c_str(),
			n,alpha,beta);
	if(inputs.size() == 1){
		printf("\tinput: ");
		inputs[0].show();
	}
	if(outputs.size() == 1){
		printf("\toutput: ");
		outputs[0].show();
	}

}
void LRNLayer::setup_shape(){
	if( (inputs.size()!=1) || (input_difs.size()!=1)){
		printf("error: lrn input blob number should be 1\n");
		exit(0);
	}

	// output
	const Blob& ib = inputs[0];
	outputs.resize(1);
	output_difs.resize(1);
	outputs[0].set_shape(ib.n, ib.c, ib.h, ib.w);
	output_difs[0].set_shape(outputs[0]);
}
void LRNLayer::setup_data(){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: lrn output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	output_difs[0].alloc();
}
