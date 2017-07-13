#include "stdlib.h"
#include "Layers.h"
#include "DataLayer.h"
#include "ConvLayer.h"
#include "PoolLayer.h"
#include "LRNLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"
#include "ConcatLayer.h"

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
void Layers::add_data(const std::string& name, int n, int c, int h, int w, const std::string& method){
	Layer* l = new DataLayer(name, n,c,h,w, method);
	add(name, &l);
}
void Layers::add_conv(const std::string&name, const std::vector<int>& p4, const std::string& activity){
	Layer* l = new ConvLayer(name,p4[0],p4[1],p4[2],p4[3],activity);
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
void Layers::add_fc(const std::string&name, int n, const std::string& activity){
	Layer* l = new FCLayer(name,n,activity);
	add(name, &l);
}
void Layers::add_loss(const std::string&name, const std::string& method){
	Layer* l = new LossLayer(name,method);
	add(name, &l);
}
void Layers::add_concat(const std::string& name){
	Layer* l = new ConcatLayer(name);
	add(name, &l);
}
