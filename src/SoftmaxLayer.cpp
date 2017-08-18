#include <cfloat>
#include "stdlib.h"
#include "SoftmaxLayer.h"
SoftmaxLayer::SoftmaxLayer(const std::string&name):
				Layer(name,"softmax"){
}
SoftmaxLayer::~SoftmaxLayer(){
}
void SoftmaxLayer::forward(){
	//printf("forward: %s\n",type.c_str());
	//show_inputs();

	int batch_size = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();
	float *pdata = inputs[0].data;
	float *maxdata = maxs.data, *sumdata = sums.data, *softmaxdata = softmaxblob.data;
	float *odata = outputs[0].data;

	float loss = 0.0f;
	for(int i=0;i<batch_size;++i){
		for(int k=0;k<hw;++k){
			maxdata[k] = pdata[k];
			for(int j=1;j<c;++j)
				maxdata[k] = std::max(maxdata[k],pdata[j*hw+k]);
		}

		for(int j=0;j<c;++j)
			for(int k=0;k<hw;++k)
				softmaxdata[j*hw+k] = exp(pdata[j*hw+k] - maxdata[k]);

		for(int k=0;k<hw;++k){
			sumdata[k] = softmaxdata[k];
			for(int j=1;j<c;++j)
				sumdata[k] += softmaxdata[j*hw + k];
		}

		for(int j=0;j<c;++j)
			for(int k=0;k<hw;++k){
				softmaxdata[j*hw +k] /= sumdata[k];
				odata[i*hw + k] = softmaxdata[j*hw + k];
			}

		pdata += c*hw;
		odata += c*hw;
	}

	//show_outputs();
	
}
void SoftmaxLayer::backward(){
	printf("backward: %s\n",type.c_str());
}
void SoftmaxLayer::show()const {
	printf("[%s] name: %s\n", type.c_str(), name.c_str());
}
void SoftmaxLayer::setup_shape(){
	if(inputs.size()!=1){
		printf("error: softmax input blob number should be 1\n");
		exit(0);
	}
	
	outputs.resize(1);
	outputs[0].set_shape(inputs[0]);

	maxs.set_shape(1,1,inputs[0].h,inputs[0].w);
	sums.set_shape(1,1,inputs[0].h,inputs[0].w);
	softmaxblob.set_shape(1,inputs[0].c,inputs[0].h,inputs[0].w);
}
void SoftmaxLayer::setup_data(){
	if(outputs.size()!=1){
		printf("error: softmax output blob number should be 1\n");
		exit(0);
	}
	outputs[0].alloc();
	maxs.alloc();
	sums.alloc();
	softmaxblob.alloc();
}
void SoftmaxLayer::setup_dif_shape(){
	if(input_difs.size()!=1){
		printf("error: softmax input blob number should be 1\n");
		exit(0);
	}
	output_difs.resize(1);
	output_difs[0].set_shape(outputs[0]);
}
void SoftmaxLayer::setup_dif_data(){
	if(output_difs.size()!=1){
		printf("error: softmax output blob number should be 1\n");
		exit(0);
	}
	output_difs[0].alloc();
}

