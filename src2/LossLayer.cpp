#include "LossLayer.h"
LossLayer::LossLayer(){}
LossLayer::LossLayer(const LayerUnit& u):Layer(u){
	u.geta("method",method);
}
void LossLayer::show(){
	Layer::show();
	printf("  (method) %s\n",method.c_str());
}
void LossLayer::setup_outputs(){
	inplace = false;
	outputs[0].set_shape(1,1,1,1);
	outputs[0].alloc();
}
