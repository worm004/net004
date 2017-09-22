#include "stdlib.h"
#include "SplitLayer.h"
SplitLayer::SplitLayer(){}
SplitLayer::SplitLayer(const JsonValue& j):Layer(j){
}

void SplitLayer::show(){
	Layer::show();
}
void SplitLayer::setup_outputs(){
	outputs[0].set_shape(inputs[0]);
	setup_outputs_data();
}
void SplitLayer::forward(){
	//show_outputs();
}
