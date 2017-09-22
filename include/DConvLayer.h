#ifndef DCONVLAYER_H
#define DCONVLAYER_H
#include "BaseLayer.h"
class DConvLayer: public Layer{
	public:
	DConvLayer();
	DConvLayer(const JsonValue& j);
	virtual void show();
	virtual void setup_outputs();
	virtual void forward();
	bool bias;
	int group, 
	    num,
	    kernel_size_h, kernel_size_w, 
	    pad_h, pad_w, 
	    stride_h, stride_w;
	 private:
	 float * col = 0;
};
#endif
