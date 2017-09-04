#ifndef EltwiseLAYER_H
#define EltwiseLAYER_H
#include "BaseLayer.h"
class EltwiseLayer: public Layer{
	public:
	EltwiseLayer();
	EltwiseLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void forward_sum();
	std::string method;
	float coef0, coef1;

	typedef void (EltwiseLayer::*FORWARD_FUNC) ();
	FORWARD_FUNC f = 0;
};
#endif

