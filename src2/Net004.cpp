#include "stdlib.h"
#include "Net004.h"
#include "Parser.h"
using namespace std;
void Net004::load(const std::string& net_path, const std::string& model_path){
	NetParser parser;
	parser.read_net(net_path);
	is_train = parser.get_net_mode();
	name = parser.get_net_name();
	const std::vector<LayerUnit>& layers = parser.get_layers();
	for(int i=0;i<layers.size();++i) ls.add(layers[i]);
	ls.init();
	ls.show();
}
Layer* Net004::operator [](const std::string& name){
	return ls[name];
}
void Net004::pre_alloc(){
}
void Net004::forward(){
}
