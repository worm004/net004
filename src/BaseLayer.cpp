#include "stdlib.h"
#include "BaseLayer.h"
using namespace std;
int i2o_floor(int w, int kernel, int stride, int padding){
	return (w + 2 * padding - kernel) / stride + 1;
}
int i2o_ceil(int w, int kernel, int stride, int padding){
	return (w + 2 * padding - kernel + stride - 1) / stride + 1;
}
Layer::Layer(){
}
Layer::Layer(const JsonValue& j){
	j_ = j;
	const JsonValue& attrs = j.jobj.at("attrs");
	type = attrs.jobj.at("type").jv.s;
	name = attrs.jobj.at("name").jv.s;
	if(j.jobj.find("params")!=j.jobj.end())
	for(const auto &i : j.jobj.at("params").jobj){
		if(params.find(i.first) != params.end()){
			printf("[%s %s] duplicate params: %s\n",type.c_str(), name.c_str(),i.first.c_str());
			exit(0);
		}
		int n = i.second.jarray[0].jv.d;
		int c = i.second.jarray[1].jv.d;
		int h = i.second.jarray[2].jv.d;
		int w = i.second.jarray[3].jv.d;
		params[i.first];
		params[i.first].set_shape(n,c,h,w);
		params[i.first].alloc();
	}
	if(j.jobj.find("inputs")!=j.jobj.end())
		inputs.resize(j.jobj.at("inputs").jobj.size());
	outputs.resize(1);
}
Layer::~Layer(){
	for(int i=0;i<inputs.size();++i)
		inputs[i].clear();
	for(int i=0;i<outputs.size();++i)
		outputs[i].clear();
	for(auto& i:params)
		i.second.clear();
	inputs.clear();
	outputs.clear();
	params.clear();
}
void Layer::set_inplace(bool inplace){
	this->inplace = inplace;
}
void Layer::show(){
	printf("(type name) %s %s\n",type.c_str(),name.c_str());
	printf("  (inplace) %d\n",int(inplace));
	for(const auto&i : params)
		printf("  (learnt param) %s [%d %d %d %d]\n",i.first.c_str(),i.second.n,i.second.c,i.second.h,i.second.w);
	//for(const auto&i : u.inputs)
	//	printf("  (inputs) %s %d [%d %d %d %d]\n",i.first.c_str(),i.second,inputs[i.second].n,inputs[i.second].c,inputs[i.second].h,inputs[i.second].w);
	printf("  (outputs) [%d %d %d %d]\n",outputs[0].n,outputs[0].c,outputs[0].h,outputs[0].w);
}
void Layer::show_inputs(){
	printf("input %s %s:\n",type.c_str(),name.c_str());
	for(int index = 0; index < inputs.size(); ++index){
		printf("[index] %d\n",index);
		Blob &input = inputs[index];
		int n = input.n, chw = input.chw();
		for(int b=0;b<n;++b){
			printf("[batch] %d\n",b);
			for(int k=0;k<chw;++k)
				printf("%g ", input.data[b*chw + k]);
			printf("\n");
		}
		printf("\n");
	}
}
void Layer::show_outputs(){
	printf("output %s %s\n",type.c_str(),name.c_str());
	for(int index = 0; index < outputs.size(); ++index){
		printf("[index] %d\n",index);
		Blob &output = outputs[index];
		int n = output.n, chw = output.chw();
		for(int b=0;b<n;++b){
			printf("[batch] %d\n",b);
			for(int k=0;k<chw;++k)
				printf("%g ", output.data[b*chw + k]);
			printf("\n");
		}
		printf("\n");
	}
}
void Layer::setup_outputs_data(){
	if(inplace && (outputs[0].nchw() == inputs[0].nchw()))
		outputs[0].set_data(inputs[0].data);
	else {
		inplace = false;
		outputs[0].alloc();
	}
}
