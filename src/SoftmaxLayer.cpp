#include <cmath>
#include "stdlib.h"
#include "SoftmaxLayer.h"
SoftmaxLayer::SoftmaxLayer(){}
SoftmaxLayer::SoftmaxLayer(const JsonValue& j):Layer(j){
}
void SoftmaxLayer::show(){
	Layer::show();
}
void SoftmaxLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	setup_outputs_data();

	Blob &ib = inputs[0];
	maxs.set_shape(1,1,ib.h,ib.w);
	maxs.alloc();
	sums.set_shape(1,1,ib.h,ib.w);
	sums.alloc();
	softmaxblob.set_shape(1,ib.c,ib.h,ib.w);
	softmaxblob.alloc();
}
void SoftmaxLayer::forward(){
	//show_inputs();
	int batch_size = inputs[0].n, c = inputs[0].c, hw = inputs[0].hw();
	float *pdata = inputs[0].data;
	float *maxdata = maxs.data, *sumdata = sums.data, *softmaxdata = softmaxblob.data;
	float *odata = outputs[0].data;
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
				odata[j*hw + k] = softmaxdata[j*hw + k];
			}
		pdata += c*hw;
		odata += c*hw;
	}
	//show_outputs();
}
