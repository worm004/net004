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
	void add(const LayerUnit& u);
	void init();
	void show();

	void init_n2i();
	void init_forder();
	//void init_inplace();
	Layer* operator [](const std::string& name);

	private:
	std::vector<Layer*> layers;
	std::map<std::string, int> n2i;
	std::vector<int> forder;
	std::map<std::string, std::vector<std::string>> cs;

	typedef std::map<std::string, Layer*(*)(const LayerUnit&)> LayerTypeMap;
	LayerTypeMap layer_type_map;
};
#endif
