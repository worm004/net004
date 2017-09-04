#ifndef DATALAYER_H
#define DATALAYER_H
#include "BaseLayer.h"
class DataLayer: public Layer{
	public:
	DataLayer();
	DataLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	int n,c,h,w;
	std::string method;
};
#endif
