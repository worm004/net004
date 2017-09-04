#ifndef POOLLAYER_H
#define POOLLAYER_H
#include "BaseLayer.h"
class PoolLayer: public Layer{
	public:
	PoolLayer();
	PoolLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void forward_max();
	void forward_avg();

	bool global;
	int pad, kernel, stride;
	std::string method;
	typedef void (PoolLayer::*FORWARD_FUNC)();
	FORWARD_FUNC f;
};
#endif
