#include "stdlib.h"
#include "BaseLayer.h"
using namespace std;
ParamUnit::ParamUnit(){}
ParamUnit::ParamUnit(float v){
	type = "float";
	fval = v;
}
ParamUnit::ParamUnit(const std::string& v){
	type = "string";
	sval = v;
}
void LayerUnit::clear(){
	attrs.clear();
	params.clear();
	inputs.clear();
}
bool LayerUnit::exista(const std::string&key)const {
	return attrs.find(key) != attrs.end();
}
bool LayerUnit::existp(const std::string&key)const{
	return params.find(key) != params.end();
}
bool LayerUnit::existi(const std::string&key)const{
	return inputs.find(key) != inputs.end();
}
void LayerUnit::checka(const std::string& key, const std::string& type)const{
	if(!exista(key)){
		printf("unknown key (%s) in attrs\n",key.c_str());
		exit(0);
	}
	if(attrs.at(key).type != type){
		printf("attrs[\"%s\"]'s type (%s) should be %s\n",key.c_str(),attrs.at(key).type.c_str(), type.c_str());
		exit(0);
	}
}
void LayerUnit::geta(const std::string& key, float& val)const{
	checka(key,"float");
	val = attrs.at(key).fval;
}
void LayerUnit::geta(const std::string& key, std::string& val)const{
	checka(key,"string");
	val = attrs.at(key).sval;
}
Layer::Layer(){
}
Layer::Layer(const LayerUnit& u){
	this->u = u;
	u.geta("name",name);
	u.geta("type",type);
	for(const auto& i:u.params){
		if(params.find(i.first) != params.end()){
			printf("[%s %s] duplicate params: %s\n",type.c_str(), name.c_str(),i.first.c_str());
			exit(0);
		}
		params[i.first];
		params[i.first].set_shape(i.second[0],i.second[1],i.second[2],i.second[3]);
	}
	inputs.resize(u.inputs.size());
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
	for(const auto&i : params)
		printf("  (learnt param) %s [%d %d %d %d]\n",i.first.c_str(),i.second.n,i.second.c,i.second.h,i.second.w);
}
