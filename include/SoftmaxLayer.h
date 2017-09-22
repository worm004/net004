#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H
#include "BaseLayer.h"
class SoftmaxLayer: public Layer{
	public:
	SoftmaxLayer();
	SoftmaxLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	Blob softmaxblob, maxs, sums;
};
#endif
