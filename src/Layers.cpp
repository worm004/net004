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
#include "BNLayer.h"
#include "ScaleLayer.h"
#include "EltwiseLayer.h"
#include "ReshapeLayer.h"
#include "SoftmaxLayer.h"
#include "RoipoolLayer.h"
#include "ProposalLayer.h"
#include "DConvLayer.h"
#include "CropLayer.h"
using namespace std;

template<typename T> 
Layer* create_layer(const JsonValue& j){
	return new T(j);
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
		{"concat",&create_layer<ConcatLayer>},
		{"bn",&create_layer<BNLayer>},
		{"scale",&create_layer<ScaleLayer>},
		{"eltwise",&create_layer<EltwiseLayer>},
		{"reshape",&create_layer<ReshapeLayer>},
		{"softmax",&create_layer<SoftmaxLayer>},
		{"roipooling",&create_layer<RoipoolLayer>},
		{"proposal",&create_layer<ProposalLayer>},
		{"dconv",&create_layer<DConvLayer>},
		{"crop",&create_layer<CropLayer>}
	};
}
void Layers::add(const JsonValue& json){
	string type;
	type = json.jobj.at("attrs").jobj.at("type").jv.s;
	if(layer_type_map.find(type) == layer_type_map.end()){
		printf("unknown layer: %s\n",type.c_str());
		exit(0);
	}
	layers.push_back(layer_type_map[type](json));
	if(train) layers.back()->init_train();
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
	for(int i=0;i<layers.size();++i){
		string name = layers[i]->name;
		const vector<string>& ns = cs[name];
		if(ns.size() == 1) 
			layers[n2i[ns[0]]]->set_inplace(true);
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
		ins[name] = layers[i]->inputs.size();
		if(layers[i]->j_.jobj.find("inputs") != layers[i]->j_.jobj.end())
		for(const auto& j:layers[i]->j_.jobj.at("inputs").jobj){
			if(cs.find(j.first) == cs.end()) cs[j.first];
			cs[j.first].push_back(name);
		}
	}
	for(auto i:ins) if(i.second == 0) input_layers.push_back(i.first);

	//for(auto i:cs){
	//	printf("%s->\n",i.first.c_str());
	//	for(auto j:i.second)
	//		printf("\t%s\n",j.c_str());
	//}
	//getchar();
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
