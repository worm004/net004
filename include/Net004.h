#ifndef NET004_H
#define NET004_H
#include <string>
#include "Connections.h"
#include "Layers.h"

class Net004{
	public:
	Net004(bool is_train = false):is_train(is_train){}
	Net004(const std::string&name, bool is_train = false):name(name),is_train(is_train){};
	void check();
	void show();
	void setup();
	void forward();
	void backward();

	Connections cs;
	Layers ls;

	std::string name;
	bool is_train;
};
#endif
