#include "stdlib.h"
#include "SplitLayer.h"
SplitLayer::SplitLayer(){}
SplitLayer::SplitLayer(const LayerUnit& u):Layer(u){
}

void SplitLayer::show(){
	Layer::show();
}
void SplitLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	inplace = true;
	setup_outputs_data();
}
void SplitLayer::forward(){
}
