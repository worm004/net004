#ifndef BNLAYER_H
#define BNLAYER_H
#include "BaseLayer.h"

class BNLayer: public Layer{
	public:
	BNLayer(const std::string& name);
	virtual ~BNLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void show() const;
	virtual int parameter_number();

	Blob mean, variance, mean_dif, variance_dif, scale, scale_dif;
};
#endif
