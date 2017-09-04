#ifndef LOSSLAYER_H
#define LOSSLAYER_H
#include "BaseLayer.h"
class LossLayer: public Layer{
	public:
	LossLayer();
	LossLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void init_softmax();
	void forward_softmax();

	typedef void (LossLayer::*FUNCTION)();
	FUNCTION forward_f, init_f;
	std::string method;

	// softmax
	Blob softmaxblob, maxs, sums;
};
#endif
