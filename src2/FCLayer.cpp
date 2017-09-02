#include "FCLayer.h"
FCLayer::FCLayer(){}
FCLayer::FCLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("bias",v); bias = v;
	u.geta("num",v); num = v;
}
void FCLayer::show(){
	Layer::show();
	printf("  (bias) %d\n",int(bias));
	printf("  (num) %d\n",int(num));
}

void FCLayer::setup_outputs(){
	const Blob& ib = inputs[0];
	outputs[0].set_shape(ib.n, num, 1, 1);
	outputs[0].alloc();
	inplace = false;
}
