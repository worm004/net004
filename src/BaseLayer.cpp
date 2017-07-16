#include "stdlib.h"
#include "BaseLayer.h"
Layer::Layer(
	const std::string& name, 
	const std::string& type):
		name(name),
		type(type){
}
Layer::~Layer(){
}
void Layer::setup_shape(){
}
void Layer::setup_data(){
}
void Layer::setup(){
	setup_shape();
	setup_data();
}
void Layer::connect2(Layer& l){
	if( (outputs.size()!=1) || (output_difs.size()!=1)){
		printf("error: connnect2: output blob number should be 1\n");
		exit(0);
	}
	l.inputs.push_back(Blob());
	l.inputs.back().set_shape(outputs[0]);
	l.inputs.back().set_data(outputs[0].data);
	l.input_difs.push_back(Blob());
	l.input_difs.back().set_shape(outputs[0]);
	l.input_difs.back().set_data(output_difs[0].data);
}
int Layer::parameter_number(){
	return 0;
}
int Layer::input_parameter_number(){
	int n = 0;
	for(const auto&i: inputs)
		n += i.total();
	return n;
}
int Layer::output_parameter_number(){
	int n = 0;
	for(const auto&i: outputs)
		n += i.total();
	return n;
}
