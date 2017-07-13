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
	//if(l.name == "concat_3a") printf("connect: %s %s %p %d %d\n",l.name.c_str(),l.type.c_str(),&l,l.inputs.size(), l.input_difs.size());
}
int Layer::parameter_number(){
	return 0;
}
