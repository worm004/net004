#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H
#include "BaseLayer.h"
class SoftmaxLayer: public Layer{
	public:
	SoftmaxLayer(const std::string&name);
	virtual ~SoftmaxLayer();
	virtual void forward();
	virtual void backward();
	virtual void show()const;
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	
	Blob softmaxblob, maxs, sums;
	Blob predict, gt;
};

#endif
