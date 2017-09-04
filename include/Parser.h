#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <string>
#include <map>
#include "BaseLayer.h"

class Net004;
class NetParser{
	public:
	void read_net(const std::string& path);
	void write_net(const std::string& path);

	void add_layer(const LayerUnit& u);
	void set_net_name(const std::string& name);
	void set_net_mode(bool is_train);
	std::string get_net_name();
	bool get_net_mode();
	const std::vector<LayerUnit>& get_layers();
	
	private:
	std::vector<LayerUnit> layers;
	std::string net_name;
	bool is_train = false;
};

class ModelParser{
	public:
	ModelParser();
	void read_model(const std::string& path, Net004* net);
};
#endif
