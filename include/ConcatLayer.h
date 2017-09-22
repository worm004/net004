#ifndef CONCATLAYER_H
#define CONCATLAYER_H
#include "BaseLayer.h"
class ConcatLayer: public Layer{
	public:
	ConcatLayer();
	ConcatLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void forward_channel();
	typedef void (ConcatLayer::*FORWARD_FUNC) ();
	FORWARD_FUNC f = 0;
	std::string method;
};
#endif
