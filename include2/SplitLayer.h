#ifndef SPLITLAYER_H
#define SPLITLAYER_H
#include "BaseLayer.h"
class SplitLayer: public Layer{
	public:
	SplitLayer();
	SplitLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
};
#endif
