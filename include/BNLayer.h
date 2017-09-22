#ifndef BNLAYER_H
#define BNLAYER_H
#include "BaseLayer.h"
class BNLayer: public Layer{
	public:
	BNLayer();
	BNLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	float eps;
};
#endif
