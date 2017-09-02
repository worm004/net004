#include "ActivityLayer.h"
ActivityLayer::ActivityLayer(){}
ActivityLayer::ActivityLayer(const LayerUnit& u):Layer(u){
	u.geta("neg_slope",neg_slope);
	u.geta("method",method);
}
void ActivityLayer::show(){
	Layer::show();
	printf("  (neg_slope) %g\n",neg_slope);
	printf("  (method) %s\n",method.c_str());
}
void ActivityLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	if(inplace) outputs[0].set_data(inputs[0].data);
	outputs[0].alloc();
}
