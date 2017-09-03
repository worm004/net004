#include "stdlib.h"
#include "Net004.h"
#include "Parser.h"
using namespace std;
void Net004::load(const std::string& net_path, const std::string& model_path){
	NetParser nparser;
	nparser.read_net(net_path);
	is_train = nparser.get_net_mode();
	name = nparser.get_net_name();
	const std::vector<LayerUnit>& layers = nparser.get_layers();
	for(int i=0;i<layers.size();++i) ls.add(layers[i]);
	ls.init();
	ModelParser mparser;
	mparser.read_model(model_path, this);
}
Layer* Net004::operator [](const std::string& name){
	return ls[name];
}
Layer* Net004::operator [](int index){
	return ls[index];
}
void Net004::pre_alloc(){
	// connect
	for(int i=0;i<ls.size();++i){
		Layer* l = ls[i];
		if(l->type != "data") continue;
		l->setup_outputs();
	}
	for(int i=0;i<ls.size();++i){
		Layer* l = ls[i];
		if(l->type == "data") continue;
		for(const auto& j:l->u.inputs){
			Layer* l1 = ls[j.first];
			int index = j.second;
			//printf("%s <- %s %d\n",l->name.c_str(), l1->name.c_str(),index);
			Blob &b = l->inputs[index];
			Blob &b1 = l1->outputs[0];
			b.clear();
			b.set_shape(b1);
			b.set_data(b1.data);
		}
		l->setup_outputs();
	}
}
void Net004::forward(){
	for(int i=0;i<ls.size();++i)
		ls[i]->forward();
}
void Net004::show(){
	printf("(net) %s\n",name.c_str());
	for(int i=0;i<ls.size();++i)
		ls[i]->show();
}
