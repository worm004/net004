#include <cmath>
#include "stdlib.h"
#include "EltwiseLayer.h"
EltwiseLayer::EltwiseLayer(){}
EltwiseLayer::EltwiseLayer(const LayerUnit& u):Layer(u){
	u.geta("coeff0",coef0);
	u.geta("coeff1",coef1);
	u.geta("method",method);
	if(method == "sum") f = &EltwiseLayer::forward_sum;
	else{
		printf("not implemented method %s in eltwise layer\n",method.c_str());
		exit(0);
	}
}
void EltwiseLayer::show(){
	Layer::show();
	printf("  (method) %s\n", method.c_str());
}
void EltwiseLayer::setup_outputs(){
	if((inputs.size() != 2) || !inputs[0].is_shape_same(inputs[1])){
		printf("wrong inputs in eltwise layer\n");
		exit(0);
	}
	outputs[0].set_shape(inputs[0]);
	inplace = false;
	setup_outputs_data();
}
void EltwiseLayer::forward(){
	(this->*f)();
}
void EltwiseLayer::forward_sum(){
	int nchw = outputs[0].nchw();
	float *b0 = inputs[0].data, *b1 = inputs[1].data, *output_data = outputs[0].data;
	for(int i=0;i<nchw;++i)
		output_data[i] = b0[i]*coef0 + b1[i]*coef1;
}

