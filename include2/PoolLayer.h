#ifndef POOLLAYER_H
#define POOLLAYER_H
#include "BaseLayer.h"
class PoolLayer: public Layer{
	public:
	PoolLayer();
	PoolLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();

	bool global;
	int pad, kernel, stride;
};
#endif
