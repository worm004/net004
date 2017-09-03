#ifndef NET004_H
#define NET004_H
#include <string>
#include "Layers.h"
class Layer;
class Net004{
	public:
	void load(const std::string& net_path, const std::string& model_path);
	Layer* operator [](const std::string& name);
	Layer* operator [](int index);
	void pre_alloc();
	void forward();
	void show();

	Layers ls;
	private:
	std::string name;
	bool is_train;
};
#endif

