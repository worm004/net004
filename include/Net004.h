#ifndef NET004_H
#define NET004_H
#include <string>
#include "Connections.h"
#include "Layers.h"

class Net004{
	public:
	Net004(const std::string&name):name(name){};
	void check();
	void show();

	Connections cs;
	Layers ls;

	private:
	std::string name;
};
#endif
