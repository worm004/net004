#ifndef ELTWISELAYER_H
#define ELTWISELAYER_H
#include "BaseLayer.h"
class EltwiseLayer: public Layer{
	public:
	EltwiseLayer(const std::string& name, const std::string& method);
	virtual ~EltwiseLayer();
	virtual void forward();
	virtual void backward();
	virtual void setup_shape();
	virtual void setup_data();
	virtual void setup_dif_shape();
	virtual void setup_dif_data();
	virtual void show() const;
	virtual int parameter_number();
	
	std::string method;
};

#endif
