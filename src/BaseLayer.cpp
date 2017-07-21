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
	int sum = 0;
	for(const auto&i: inputs)
		sum += i.nchw();
	return sum;
}
int Layer::output_parameter_number(){
	int sum = 0;
	for(const auto&i: outputs)
		sum += i.nchw();
	return sum;
}
void Layer::show_inputs(){
	if(inputs.size() == 0){
		printf("\nno input\n");
		return;
	}
	for(int index = 0; index < inputs.size(); ++index){
		printf("\ninput index: %d\n",index);
		Blob &input = inputs[index];
		int n = input.n, c = input.c, h = input.h, w = input.w;

		for(int b=0;b<n;++b){
			printf("input batch %d:\n",b);
			for(int k=0;k<1/*c*/;++k){
				for(int i=0;i<h;++i){
					for(int j=0;j<w;++j)
						printf("%f ",input.data[b*c*h*w + h*w*k + i*w + j]);
					printf("\n");
				}
				printf("\n");
			}
		}
	}
}
void Layer::show_outputs(){
	if(outputs.size() == 0){
		printf("\nno output\n");
		return;
	}
	for(int index = 0; index < outputs.size(); ++index){
		printf("\noutput index: %d\n",index);
		Blob &output = outputs[index];
		int n = output.n, c = output.c, h = output.h, w = output.w;

		for(int b=0;b<n;++b){
			printf("output batch %d:\n",b);
			for(int k=0;k<1/*c*/;++k){
				for(int i=0;i<h;++i){
					for(int j=0;j<w;++j)
						printf("%f ",output.data[b*c*h*w + h*w*k + i*w + j]);
					printf("\n");
				}
				printf("\n");
			}
		}
	}
}
