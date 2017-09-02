#ifndef LOSSLAYER_H
#define LOSSLAYER_H
#include "BaseLayer.h"
class LossLayer: public Layer{
	public:
	LossLayer();
	LossLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();

	std::string method;
};
#endif
