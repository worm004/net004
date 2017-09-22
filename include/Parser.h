#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <string>
#include <map>
#include "BaseLayer.h"
#include "BaseRun.h"
class NetParser{
	public:
	void read_net(const std::string& path);
	void write_net(const std::string& path);

	void add_layer(const LayerUnit& u);
	void set_net_name(const std::string& name);
	std::string get_net_name();
	const std::vector<LayerUnit>& get_layers();
	
	private:
	std::vector<LayerUnit> layers;
	std::string net_name;
};
class Net004;
class ModelParser{
	public:
	ModelParser();
	void read_model(const std::string& path, Net004* net);
};
#endif
