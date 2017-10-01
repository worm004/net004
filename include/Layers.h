#ifndef LAYERS_H
#define LAYERS_H
#include <string>
#include <vector>
#include <map>
#include "BaseLayer.h"


class Layer;
class Layers{
	public:
	Layers();
	void add(const JsonValue& json);
	void init();
	void show();

	void init_n2i();
	void init_forder();
	void init_inplace();
	int size();
	Layer* operator [](const std::string& name);
	Layer* operator [](int index);//order

	std::vector<Layer*> layers;
	std::map<std::string, int> n2i;
	std::vector<int> forder;
	std::map<std::string, std::vector<std::string>> cs;

	std::vector<std::string> input_layers;
	bool train = false;

	private:
	typedef std::map<std::string, Layer*(*)(const JsonValue&)> LayerTypeMap;
	LayerTypeMap layer_type_map;
};
#endif

