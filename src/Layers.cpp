#include "stdlib.h"
#include "Layers.h"
#include "DataLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "LRNLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"
#include "ConcatLayer.h"
#include "ActivityLayer.h"
#include "SplitLayer.h"
#include "BNLayer.h"
#include "ScaleLayer.h"
#include "EltwiseLayer.h"

void Layers::add(const std::string& name, Layer** p){
	if(layers.find(name) != layers.end()){
		printf("error: duplicate name: %s\n",name.c_str());
		exit(0);
	}
	layers[name] = *p;
}

void Layers::clear(){
	for(auto i: layers){
		if (i.second){
			delete i.second;
		}
	}
	layers.clear();
}

bool Layers::exist(const std::string& name){
	return layers.find(name) != layers.end();
}
int Layers::count(){
	return layers.size();
}
Layer* Layers::operator [] (const std::string& name){
	if(!exist(name)){
		printf("error: no such layer named %s\n",name.c_str());
		exit(0);
	}
	return layers[name];
}

int Layers::parameter_number(const std::string& name){
	if(!exist(name)){
		printf("error: no such layer named %s\n",name.c_str());
		exit(0);
	}
	return layers[name]->parameter_number();
}
int Layers::input_parameter_number(const std::string& name){
	if(!exist(name)){
		printf("error: no such layer named %s\n",name.c_str());
		exit(0);
	}
	return layers[name]->input_parameter_number();
}
int Layers::output_parameter_number(const std::string& name){
	if(!exist(name)){
		printf("error: no such layer named %s\n",name.c_str());
		exit(0);
	}
	return layers[name]->output_parameter_number();
}
void Layers::show(const std::string& name){
	if(!exist(name)){
		printf("error: no such layer named %s\n",name.c_str());
		exit(0);
	}
	layers[name]->show();
}
void Layers::show(){
	printf("Layers:\n");
	for(const auto&i:layers) i.second->show();
}

// different types of layers
void Layers::add_activity(const std::string& name, const std::string& method){
	Layer* l = new ActivityLayer(name,method);
	add(name, &l);
}
void Layers::add_data(const std::string& name, int n, int c, int h, int w, const std::string& method){
	Layer* l = new DataLayer(name, n,c,h,w, method);
	add(name, &l);
}
void Layers::add_conv(const std::string&name, const std::vector<int>& p8, bool is_bias, const std::string& activity){
	Layer* l = new ConvLayer(name,p8[0],p8[1],p8[2],p8[3],p8[4],p8[5],p8[6],p8[7],is_bias,activity);
	add(name, &l);
}
void Layers::add_pool(const std::string&name, const std::vector<int>& p3, const std::string& method){
	Layer* l = new PoolLayer(name,p3[0],p3[1],p3[2],method);
	add(name, &l);
}
void Layers::add_lrn(const std::string&name, int n, float alpha,float beta){
	Layer* l = new LRNLayer(name,n,alpha,beta);
	add(name, &l);
}
void Layers::add_fc(const std::string&name, int n, bool is_bias, const std::string& activity){
	Layer* l = new FCLayer(name,n,is_bias,activity);
	add(name, &l);
}
void Layers::add_loss(const std::string&name, const std::string& method){
	Layer* l = new LossLayer(name,method);
	add(name, &l);
}
void Layers::add_concat(const std::string& name, const std::vector<std::string>& names, const std::string& method){
	Layer* l = new ConcatLayer(name, names, method);
	add(name, &l);
}
void Layers::add_split(const std::string& name){
	Layer* l = new SplitLayer(name);
	add(name, &l);
}
void Layers::add_bn(const std::string& name,float eps){
	Layer* l = new BNLayer(name, eps);
	add(name, &l);
}
void Layers::add_scale(const std::string& name,bool is_bias){
	Layer* l = new ScaleLayer(name,is_bias);
	add(name, &l);
}
void Layers::add_eltwise(const std::string& name, const std::string& method){
	Layer* l = new EltwiseLayer(name, method);
	add(name, &l);
}
