#include "DataLayer.h"
DataLayer::DataLayer(){}
DataLayer::DataLayer(const JsonValue& j):Layer(j){
	const JsonValue& attrs = j.jobj.at("attrs");
	n = attrs.jobj.at("n").jv.d;
	c = attrs.jobj.at("c").jv.d;
	h = attrs.jobj.at("h").jv.d;
	w = attrs.jobj.at("w").jv.d;
	method = attrs.jobj.at("method").jv.s;
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
