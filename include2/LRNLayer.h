#ifndef LRNLAYER_H
#define LRNLAYER_H
#include "BaseLayer.h"
class LRNLayer: public Layer{
	public:
	LRNLayer();
	LRNLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	float alpha, beta;
	int local_size;

	Blob buffer;
};
#endif

