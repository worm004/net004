#ifndef SCALELAYER_H
#define SCALELAYER_H
#include "BaseLayer.h"
class ScaleLayer: public Layer{
	public:
	ScaleLayer();
	ScaleLayer(const LayerUnit& u);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	void forward_bias();
	void forward_nbias();
	bool bias;

	typedef void (ScaleLayer::*FORWARD_FUNC) ();
	FORWARD_FUNC f;
};
#endif

