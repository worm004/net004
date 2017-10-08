#ifndef POOLLAYER_H
#define POOLLAYER_H
#include "BaseLayer.h"
class PoolLayer: public Layer{
	public:
	PoolLayer();
	virtual ~PoolLayer();
	PoolLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	virtual void backward();
	void forward_max();
	void forward_avg();
	void backward_max();
	void backward_avg();

	bool global;
	int pad, kernel, stride;
	std::string method;
	typedef void (PoolLayer::*FORWARD_FUNC)();
	FORWARD_FUNC f, bf;

	int *bp_map = 0;
};
#endif
