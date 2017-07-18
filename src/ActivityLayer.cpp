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
void ActivityLayer::forward(){
	printf("forward: %s %s\n",type.c_str(), name.c_str());
	//printf("input:\n");
	//for(int k=0;k<1;++k){
	//	for(int i=0;i<2;++i){
	//		for(int j=0;j<inputs[0].w;++j)
	//			printf("%f ",inputs[0].data[inputs[0].h * inputs[0].w *k + i*inputs[0].w + j]);
	//	}
	//}

	int n = inputs[0].total();
	for(int i=0;i<n;++i)
		if (inputs[0].data[i] < 0) outputs[0].data[i] = 0;

	//printf("\noutput:\n");
	//for(int k=0;k<1;++k){
	//	for(int i=0;i<2;++i){
	//		for(int j=0;j<outputs[0].w;++j)
	//			printf("%f ",outputs[0].data[outputs[0].h * outputs[0].w *k + i*outputs[0].w + j]);
	//	}
	//}
	//getchar();
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
