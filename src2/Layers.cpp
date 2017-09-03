#include <set>
#include "stdlib.h"
#include "Layers.h"
#include "ConvLayer.h"
#include "FCLayer.h"
#include "LossLayer.h"
#include "ActivityLayer.h"
#include "DataLayer.h"
#include "PoolLayer.h"
#include "LRNLayer.h"
#include "SplitLayer.h"
#include "ConcatLayer.h"
using namespace std;

template<typename T> 
Layer* create_layer(const LayerUnit& u){
	return new T(u);
}

Layers::Layers(){
	layer_type_map = {
		{"conv",&create_layer<ConvLayer>},
		{"data",&create_layer<DataLayer>},
		{"fc",&create_layer<FCLayer>},
		{"activity",&create_layer<ActivityLayer>},
		{"loss",&create_layer<LossLayer>},
		{"pool",&create_layer<PoolLayer>},
		{"lrn",&create_layer<LRNLayer>},
		{"split",&create_layer<SplitLayer>},
		{"concat",&create_layer<ConcatLayer>}
	};
}
void Layers::add(const LayerUnit& u){
	string type;
	u.geta("type",type);
	if(layer_type_map.find(type) == layer_type_map.end()){
		printf("unknown layer: %s\n",type.c_str());
		exit(0);
	}
	layers.push_back(layer_type_map[type](u));
}
void Layers::show(){
	for(int i=0;i<layers.size();++i)
		layers[i]->show();
}
void Layers::init(){
	init_n2i();
	init_forder();
	init_inplace();
}
void Layers::init_inplace(){
	//TODO
	for(int i=0;i<layers.size();++i){
		string name = layers[i]->name;
		//layers[i]->set_inplace((cs[name].size() == 1) && (layers[i]->inputs.size() == 1));
		layers[i]->set_inplace(false);
	}
}
void Layers::init_n2i(){
	n2i.clear();
	for(int i=0;i<layers.size();++i) n2i[layers[i]->name] = i;
}
void Layers::init_forder(){
	forder.clear();
	cs.clear();
	map<string, int> ins;
	for(int i=0;i<layers.size();++i){
		string name = layers[i]->name;
		ins[name] = layers[i]->u.inputs.size();
		for(const auto& j:layers[i]->u.inputs){
			if(cs.find(j.first) == cs.end()) cs[j.first];
			cs[j.first].push_back(name);
		}
	}
	set<string> noins;
	for(const auto& l: ins)
		if(l.second == 0) noins.insert(l.first);
	while(!noins.empty()){
		const string l = *noins.begin();
		forder.push_back(n2i[l]);
		noins.erase(noins.begin());
		if(n2i.find(l) == n2i.end()) continue;
		for(const auto& to : cs[l]){
			ins[to] -= 1;
			if(ins[to] == 0) noins.insert(to);
		}
	}
	for(const auto&l:ins){
		if(l.second ==0) continue;
		printf("layer connection error\n");
		exit(0);
	}
}
Layer* Layers::operator [](const std::string& name){
	if(n2i.find(name) == n2i.end()){
		printf("unknown layer name: %s\n",name.c_str());
		exit(0);
	}
	return layers[n2i[name]];
}
Layer* Layers::operator [](int index){
	if((index < 0) || (index >= layers.size())){
		printf("index error\n");
		exit(0);
	}
	return layers[forder[index]];
}
int Layers::size(){
	return layers.size();
}
