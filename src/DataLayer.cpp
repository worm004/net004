#include "DataLayer.h"
DataLayer::DataLayer(){}
DataLayer::DataLayer(const LayerUnit& u):Layer(u){
	float v;	
	u.geta("n",v); n = v;
	u.geta("c",v); c = v;
	u.geta("h",v); h = v;
	u.geta("w",v); w = v;
	u.geta("method",method);
}
void DataLayer::show(){
	Layer::show();
	printf("  (method) %s\n",method.c_str());
	printf("  (shape) [%d %d %d %d]\n",n,c,h,w);
}
void DataLayer::setup_outputs(){
	outputs[0].set_shape(n,c,h,w);
	inplace = false;
	setup_outputs_data();
}
void DataLayer::forward(){
	//show_outputs();
}
